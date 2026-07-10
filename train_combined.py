"""
=============================================================================
摔倒检测 - 合并数据集训练脚本
功能：
1. 合并 UR-Fall + GMDCSA24 特征
2. 应用类不平衡修正：SMOTE + Focal Loss + 加权采样 + MixUp
3. 使用 BiLSTM + BiGRU + Additive Attention 架构
4. 生成详细评估报告

类不平衡修正组合策略：
- SMOTE: 少数类插值过采样
- MixUp: 两样本线性混合增强
- WeightedRandomSampler: 类别权重采样
- Focal Loss: 降低易分类样本权重
=============================================================================
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============ 配置 ============
BASE_DIR = Path(r"D:\识别人类摔倒的智能系统")
FEATURES_DIR = BASE_DIR / "features"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

# 模型超参数
INPUT_SIZE = 10
HIDDEN_SIZE = 128
NUM_CLASSES = 2
DROPOUT = 0.3

# 训练超参数
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
EPOCHS = 150
PATIENCE = 20

# 类不平衡策略配置
USE_SMOTE = True
SMOTE_RATIO = '1:1'       # '1:1' 平衡，'1:2' 2倍
USE_MIXUP = True
MIXUP_ALPHA = 0.3          # MixUp 插值强度
USE_WEIGHTED_SAMPLER = True
USE_FOCAL_LOSS = True
FOCAL_ALPHA = 2.0
FOCAL_GAMMA = 2.0

# 验证比例（从合并数据中划分）
VAL_RATIO = 0.15
TEST_RATIO = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============ MixUp 数据增强 ============
def mixup_data(x, y, alpha=0.3):
    """MixUp: 两样本线性插值"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp 损失"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============ Focal Loss ============
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal.mean()


# ============ 加权交叉熵 ============
class WeightedBCEWithLogits(nn.Module):
    """带类别权重的交叉熵"""
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        return F.binary_cross_entropy_with_logits(
            inputs[:, 1] - inputs[:, 0],  # 转二分类logits
            targets.float(),
            pos_weight=self.pos_weight
        )


# ============ SMOTE 过采样 ============
def apply_smote(X, y, ratio='1:1'):
    """SMOTE 过采样少数类"""
    try:
        from imblearn.over_sampling import SMOTE
        n_normal = int(np.sum(y == 0))
        n_fall = int(np.sum(y == 1))
        print(f"  [SMOTE] 原始: Normal={n_normal}, Fall={n_fall}")

        if ratio == '1:1':
            target_fall = n_normal
        elif ratio == '1:2':
            target_fall = min(n_fall * 2, n_normal)
        else:
            target_fall = n_normal

        k = min(5, n_fall - 1) if n_fall > 1 else 1
        sm = SMOTE(random_state=42, k_neighbors=k,
                   sampling_strategy={0: n_normal, 1: target_fall})
        X_flat = X.reshape(len(X), -1)
        X_res, y_res = sm.fit_resample(X_flat, y)
        X_res = X_res.reshape(-1, X.shape[1], X.shape[2])
        print(f"  [SMOTE] 过采样后: Normal={int(np.sum(y_res==0))}, Fall={int(np.sum(y_res==1))}")
        return X_res, y_res
    except ImportError:
        print("  [警告] imblearn 未安装，跳过 SMOTE")
        return X, y


# ============ 数据集类 ============
class CombinedDataset(Dataset):
    def __init__(self, X, y, augmentation=True, noise_scale=0.03):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augmentation = augmentation
        self.noise_scale = noise_scale

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        if self.augmentation:
            # 高斯噪声
            if np.random.rand() < 0.3:
                x = x + torch.randn_like(x) * self.noise_scale
            # 时间掩码
            if np.random.rand() < 0.2:
                mask_len = min(np.random.randint(1, 4), len(x) - 1)
                start = np.random.randint(0, len(x) - mask_len)
                x[start:start + mask_len] = 0
            # 特征抖动
            if np.random.rand() < 0.2:
                scale = np.random.uniform(0.92, 1.08, size=(x.shape[1],))
                x = x * torch.FloatTensor(scale)
        return x, y


# ============ 模型定义 ============
class BiLSTMAttention(nn.Module):
    """
    BiLSTM + BiGRU + Additive Attention
    对应 Nature 2025 消融实验最优架构
    """

    def __init__(self, input_size, hidden_size, num_classes, dropout=0.3):
        super().__init__()
        # BiLSTM
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=2, batch_first=True,
                            bidirectional=True, dropout=dropout)
        # BiGRU
        self.gru = nn.GRU(hidden_size * 2, hidden_size,
                          num_layers=1, batch_first=True,
                          bidirectional=True)
        # Additive Attention（与 Nature 2025 一致）
        self.att_W = nn.Linear(hidden_size * 2, hidden_size)
        self.att_v = nn.Linear(hidden_size, 1)
        # 全连接
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

    def additive_attention(self, lstm_out):
        """加性注意力（Additive Attention / Bahdanau Attention）"""
        # score = v^T * tanh(W * h)
        energy = torch.tanh(self.att_W(lstm_out))          # (B, T, hidden*2) -> (B, T, hidden)
        scores = self.att_v(energy).squeeze(-1)              # (B, T)
        att_weights = F.softmax(scores, dim=1)               # (B, T)
        context = torch.bmm(att_weights.unsqueeze(1), lstm_out)  # (B, 1, hidden*2)
        return context.squeeze(1), att_weights

    def forward(self, x):
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (B, T, hidden*2)
        # BiGRU
        gru_out, _ = self.gru(lstm_out)  # (B, T, hidden*2)
        # Additive Attention
        context, _ = self.additive_attention(gru_out)
        # 分类
        out = self.fc(self.dropout(context))
        return out


# ============ 评估器 ============
class Evaluator:
    @staticmethod
    def find_best_threshold(y_true, y_prob):
        best_f1, best_thresh = 0, 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            preds = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t
        return best_thresh, best_f1

    @staticmethod
    def evaluate(model, X, y, threshold=0.5, device='cpu'):
        model.eval()
        preds, probs = [], []
        with torch.no_grad():
            for i in range(0, len(X), BATCH_SIZE):
                batch = torch.FloatTensor(X[i:i + BATCH_SIZE]).to(device)
                out = model(batch)
                p = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                preds.extend((p >= threshold).astype(int))
                probs.extend(p)
        return np.array(preds), np.array(probs), y

    @staticmethod
    def full_report(y_true, y_prob, threshold):
        preds = (y_prob >= threshold).astype(int)
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        cm = confusion_matrix(y_true, preds)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp = cm[0]
            fn, tp = cm[1] if len(cm) > 1 else (0, 0)

        print(f"""
╔══════════════════════════════════════════════════╗
║           合并数据集测试报告（阈值={threshold:.2f}）         ║
╠══════════════════════════════════════════════════╣
║  准确率 (Accuracy):  {acc:.4f} ({acc*100:.1f}%)               ║
║  精确率 (Precision): {prec:.4f} ({prec*100:.1f}%)               ║
║  召回率 (Recall):    {rec:.4f} ({rec*100:.1f}%)               ║
║  F1分数 (F1-Score):  {f1:.4f} ({f1*100:.1f}%)               ║
╠══════════════════════════════════════════════════╣
║  混淆矩阵:                                    ║
║                  预测Normal    预测Fall         ║
║  实际Normal:         {tn:4d}        {fp:4d}            ║
║  实际Fall:           {fn:4d}        {tp:4d}            ║
╚══════════════════════════════════════════════════╝""")
        print(classification_report(y_true, preds, target_names=['Normal', 'Fall']))
        return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
                'cm': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}}

    @staticmethod
    def plot(y_true, y_prob, save_dir):
        save_dir = Path(save_dir)
        best_thresh, _ = Evaluator.find_best_threshold(y_true, y_prob)
        preds = (y_prob >= best_thresh).astype(int)
        cm = confusion_matrix(y_true, preds)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Fall'],
                    yticklabels=['Normal', 'Fall'], ax=axes[0],
                    annot_kws={'size': 16})
        axes[0].set_title(f'Confusion Matrix (t={best_thresh:.2f})')
        axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

        axes[1].plot(fpr, tpr, lw=2, label=f'AUC={roc_auc:.3f}')
        axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[1].set_title('ROC Curve'); axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].hist(y_prob[y_true == 0], bins=30, alpha=0.7, label='Normal', color='green')
        axes[2].hist(y_prob[y_true == 1], bins=30, alpha=0.7, label='Fall', color='red')
        axes[2].axvline(best_thresh, color='blue', ls='--')
        axes[2].set_title('Probability Distribution')
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        path = save_dir / "combined_metrics.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return roc_auc, best_thresh


# ============ 训练器 ============
class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion,
                 optimizer, scheduler, config):
        self.model = model.to(config['device'])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.best_val_f1 = 0
        self.patience_counter = 0
        self.history = {'train_loss': [], 'train_acc': [], 'train_f1': [],
                       'val_loss': [], 'val_acc': [], 'val_f1': [],
                       'val_prec': [], 'val_rec': []}

    def train_epoch(self, use_mixup=False, mixup_alpha=0.3):
        self.model.train()
        total_loss, all_preds, all_labels = 0, [], []
        for X, y in self.train_loader:
            X, y = X.to(self.config['device']), y.to(self.config['device'])
            self.optimizer.zero_grad()

            if use_mixup and np.random.rand() < 0.5:
                X_mix, y_a, y_b, lam = mixup_data(X, y, mixup_alpha)
                out = self.model(X_mix)
                loss = mixup_criterion(self.criterion, out, y_a, y_b, lam)
            else:
                out = self.model(X)
                loss = self.criterion(out, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        avg_loss = total_loss / len(self.train_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return avg_loss, acc, f1

    def validate(self):
        self.model.eval()
        total_loss, all_preds, all_labels, all_probs = 0, [], [], []
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.config['device']), y.to(self.config['device'])
                out = self.model(X)
                loss = self.criterion(out, y)
                total_loss += loss.item()
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                preds = np.argmax(out.cpu().numpy(), axis=1)
                all_preds.extend(preds)
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(probs)

        avg_loss = total_loss / len(self.val_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        _, best_f1_binary = Evaluator.find_best_threshold(np.array(all_labels), np.array(all_probs))
        return avg_loss, acc, f1, prec, rec, best_f1_binary

    def train(self, epochs, checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print("合并数据集训练")
        print(f"{'='*60}")
        print(f"设备: {self.config['device']}")
        print(f"类不平衡策略: SMOTE={USE_SMOTE}, MixUp={USE_MIXUP}, "
              f"WeightedSampler={USE_WEIGHTED_SAMPLER}, Focal={USE_FOCAL_LOSS}")
        print(f"{'='*60}")

        for epoch in range(epochs):
            tl, ta, tf = self.train_epoch(USE_MIXUP, MIXUP_ALPHA)
            vl, va, vf, vp, vr, vf_bin = self.validate()
            self.scheduler.step()

            self.history['train_loss'].append(tl)
            self.history['train_acc'].append(ta)
            self.history['train_f1'].append(tf)
            self.history['val_loss'].append(vl)
            self.history['val_acc'].append(va)
            self.history['val_f1'].append(vf)
            self.history['val_prec'].append(vp)
            self.history['val_rec'].append(vr)

            lr = self.optimizer.param_groups[0]['lr']
            print(f"E{epoch+1:3d}/{epochs} | "
                  f"TL={tl:.4f} TA={ta:.3f} | "
                  f"VL={vl:.4f} VA={va:.3f} VF1={vf:.3f} "
                  f"P={vp:.3f} R={vr:.3f} | LR={lr:.2e}")

            if vf_bin > self.best_val_f1:
                self.best_val_f1 = vf_bin
                self.patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_f1': vf_bin,
                    'config': {'input_size': INPUT_SIZE, 'hidden_size': HIDDEN_SIZE,
                               'num_classes': NUM_CLASSES}
                }, checkpoint_dir / "best_model.pth")
                print(f"  ★ 新最佳! F1={vf_bin:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= PATIENCE:
                    print(f"\n早停 (连续{PATIENCE}轮未提升)")
                    break

        return self.history


def main():
    print("\n" + "=" * 60)
    print("摔倒检测 - 合并数据集训练")
    print("=" * 60)

    # 确保目录存在
    for d in [CHECKPOINTS_DIR, RESULTS_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ============ 1. 加载 UR-Fall 数据 ============
    print("\n[1] 加载 UR-Fall 特征...")
    ur_train_X = np.load(FEATURES_DIR / "train" / "X.npy")
    ur_train_y = np.load(FEATURES_DIR / "train" / "y.npy")
    ur_val_X = np.load(FEATURES_DIR / "val" / "X.npy")
    ur_val_y = np.load(FEATURES_DIR / "val" / "y.npy")
    ur_test_X = np.load(FEATURES_DIR / "test" / "X.npy")
    ur_test_y = np.load(FEATURES_DIR / "test" / "y.npy")

    print(f"  UR-Fall 训练: {ur_train_X.shape[0]} 样本 "
          f"(ADL={np.sum(ur_train_y==0)}, Fall={np.sum(ur_train_y==1)})")
    print(f"  UR-Fall 验证: {ur_val_X.shape[0]} 样本 "
          f"(ADL={np.sum(ur_val_y==0)}, Fall={np.sum(ur_val_y==1)})")
    print(f"  UR-Fall 测试: {ur_test_X.shape[0]} 样本 "
          f"(ADL={np.sum(ur_test_y==0)}, Fall={np.sum(ur_test_y==1)})")

    # ============ 2. 加载 GMDCSA24 数据 ============
    print("\n[2] 加载 GMDCSA24 特征...")
    gmdcsa24_dir = FEATURES_DIR / "gmdcsa24"
    if not (gmdcsa24_dir / "X.npy").exists():
        print(f"  警告: GMDCSA24 特征未就绪: {gmdcsa24_dir}")
        print(f"  请先运行: python extract_gmdcsa24.py")
        print(f"  当前跳过 GMDCSA24，仅使用 UR-Fall 数据集")

        # 仅使用 UR-Fall 数据
        train_X = ur_train_X
        train_y = ur_train_y
        val_X = ur_val_X
        val_y = ur_val_y
        test_X = ur_test_X
        test_y = ur_test_y
        has_gmdcsa24 = False
    else:
        gmdcsa24_X = np.load(gmdcsa24_dir / "X.npy")
        gmdcsa24_y = np.load(gmdcsa24_dir / "y.npy")
        print(f"  GMDCSA24: {gmdcsa24_X.shape[0]} 样本 "
              f"(ADL={np.sum(gmdcsa24_y==0)}, Fall={np.sum(gmdcsa24_y==1)})")

        # 合并：UR-Fall (train+val) + GMDCSA24
        all_X = np.vstack([ur_train_X, ur_val_X, gmdcsa24_X])
        all_y = np.hstack([ur_train_y, ur_val_y, gmdcsa24_y])

        # 重新划分 train/val/test
        print(f"\n  合并后总样本: {all_X.shape[0]} "
              f"(ADL={np.sum(all_y==0)}, Fall={np.sum(all_y==1)})")
        print(f"  类别比例: {np.sum(all_y==1)/len(all_y)*100:.1f}% Fall "
              f"(原始 UR-Fall 仅 {np.sum(ur_train_y==1)/len(ur_train_y)*100:.1f}%)")

        # 划分：70% train, 15% val, 15% test（分层）
        train_X, temp_X, train_y, temp_y = train_test_split(
            all_X, all_y, test_size=VAL_RATIO + TEST_RATIO,
            random_state=42, stratify=all_y
        )
        val_ratio_adjusted = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
        val_X, test_X, val_y, test_y = train_test_split(
            temp_X, temp_y, test_size=val_ratio_adjusted,
            random_state=42, stratify=temp_y
        )

        print(f"\n  重划分后:")
        print(f"  训练集: {len(train_X)} (ADL={np.sum(train_y==0)}, Fall={np.sum(train_y==1)})")
        print(f"  验证集: {len(val_X)} (ADL={np.sum(val_y==0)}, Fall={np.sum(val_y==1)})")
        print(f"  测试集: {len(test_X)} (ADL={np.sum(test_y==0)}, Fall={np.sum(test_y==1)})")
        has_gmdcsa24 = True

    # ============ 3. 特征标准化 ============
    print("\n[3] 特征标准化...")
    scaler = StandardScaler()
    X_flat = train_X.reshape(-1, train_X.shape[-1])
    scaler.fit(X_flat)
    train_mean, train_std = scaler.mean_, scaler.scale_

    train_X_s = scaler.transform(X_flat).reshape(train_X.shape)
    val_X_s = scaler.transform(val_X.reshape(-1, val_X.shape[-1])).reshape(val_X.shape)
    test_X_s = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    print(f"  训练集标准化完成: {train_X_s.shape}")

    # ============ 4. SMOTE 过采样 ============
    if USE_SMOTE and has_gmdcsa24:
        print("\n[4] SMOTE 过采样...")
        train_X_s, train_y = apply_smote(train_X_s, train_y, SMOTE_RATIO)
    else:
        print(f"\n[4] SMOTE: {'跳过（无GMDCSA24）' if not has_gmdcsa24 else '跳过'}")

    # ============ 5. 加权采样器 ============
    if USE_WEIGHTED_SAMPLER:
        print("\n[5] 构建加权采样器...")
        n_normal = int(np.sum(train_y == 0))
        n_fall = int(np.sum(train_y == 1))
        weights = np.where(train_y == 0,
                          1.0 / n_normal,
                          1.0 / n_fall)
        weights = weights / weights.sum()
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        print(f"  权重: Normal={1/n_normal:.4f}, Fall={1/n_fall:.4f}")
    else:
        sampler = None
        print("\n[5] 加权采样器: 跳过")

    # ============ 6. 构建模型 ============
    print("\n[6] 构建 BiLSTM+BiGRU+Additive Attention 模型...")
    model = BiLSTMAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, DROPOUT)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {param_count:,}")

    # ============ 7. 损失函数 ============
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
        print(f"\n[7] 损失函数: Focal Loss (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"\n[7] 损失函数: CrossEntropyLoss")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ============ 8. 数据加载器 ============
    train_dataset = CombinedDataset(train_X_s, train_y, augmentation=True)
    val_dataset = CombinedDataset(val_X_s, val_y, augmentation=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler if sampler else None,
                              shuffle=sampler is None)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ============ 9. 训练 ============
    print("\n[8] 开始训练...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = CHECKPOINTS_DIR / f"combined_run_{ts}"

    trainer = Trainer(model, train_loader, val_loader, criterion,
                     optimizer, scheduler,
                     {'device': DEVICE})
    history = trainer.train(EPOCHS, ckpt_dir)

    # 训练曲线
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Val')
    axes[0, 1].set_title('Accuracy'); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(history['train_f1'], label='Train')
    axes[1, 0].plot(history['val_f1'], label='Val')
    axes[1, 0].set_title('F1 Score'); axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(history['val_prec'], label='Precision')
    axes[1, 1].plot(history['val_rec'], label='Recall')
    axes[1, 1].set_title('P & R'); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"combined_training_{ts}.png", dpi=150)
    plt.close()
    print(f"  训练曲线已保存")

    # ============ 10. 测试集评估 ============
    print("\n[9] 测试集评估...")
    best_ckpt = ckpt_dir / "best_model.pth"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  加载最佳模型 (epoch {ckpt['epoch']+1}, val_f1={ckpt['val_f1']:.4f})")
    else:
        print("  使用最终模型")

    preds, probs, y_true = Evaluator.evaluate(model, test_X_s, test_y,
                                               threshold=0.5, device=DEVICE)
    roc_auc, best_thresh = Evaluator.plot(y_true, probs, RESULTS_DIR)
    metrics = Evaluator.full_report(y_true, probs, best_thresh)

    # ============ 11. 保存结果 ============
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'ur_fall_train': int(len(ur_train_X)),
            'ur_fall_test': int(len(ur_test_X)),
            'gmdcsa24_loaded': has_gmdcsa24,
            'combined_total': int(len(train_X) + len(val_X) + len(test_X)) if has_gmdcsa24 else int(len(ur_train_X) + len(ur_val_X) + len(ur_test_X)),
        },
        'imbalance_strategy': {
            'smote': USE_SMOTE,
            'mixup': USE_MIXUP,
            'weighted_sampler': USE_WEIGHTED_SAMPLER,
            'focal_loss': USE_FOCAL_LOSS
        },
        'test_metrics': {
            'accuracy': float(metrics['acc']),
            'precision': float(metrics['prec']),
            'recall': float(metrics['rec']),
            'f1_score': float(metrics['f1']),
            'auc_roc': float(roc_auc)
        },
        'confusion_matrix': {
            'tn': int(metrics['cm']['tn']),
            'fp': int(metrics['cm']['fp']),
            'fn': int(metrics['cm']['fn']),
            'tp': int(metrics['cm']['tp'])
        },
        'optimal_threshold': float(best_thresh)
    }

    with open(RESULTS_DIR / f"combined_results_{ts}.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {'input_size': INPUT_SIZE, 'hidden_size': HIDDEN_SIZE,
                   'num_classes': NUM_CLASSES},
        'threshold': float(best_thresh),
        'test_metrics': results['test_metrics'],
        'scaler_mean': train_mean,
        'scaler_std': train_std
    }, MODELS_DIR / f"combined_model_{ts}.pth")

    print(f"\n结果已保存: {RESULTS_DIR / f'combined_results_{ts}.json'}")
    print(f"模型已保存: {MODELS_DIR / f'combined_model_{ts}.pth'}")

    # ============ 12. 对比基线 ============
    print("\n" + "=" * 60)
    print("★ 结果汇总")
    print("=" * 60)
    old_acc, old_f1, old_auc = 0.7315, 0.5246, 0.8654
    print(f"  准确率:  {metrics['acc']:.2%}  (基线={old_acc:.2%}, 变化={metrics['acc']-old_acc:+.2%})")
    print(f"  精确率:  {metrics['prec']:.2%}  (基线=37.2%,  变化={metrics['prec']-0.372:+.2%})")
    print(f"  召回率:  {metrics['rec']:.2%}  (基线=88.9%,  变化={metrics['rec']-0.889:+.2%})")
    print(f"  F1分数:  {metrics['f1']:.2%}  (基线={old_f1:.2%}, 变化={metrics['f1']-old_f1:+.2%})")
    print(f"  AUC-ROC: {roc_auc:.4f}   (基线={old_auc:.4f}, 变化={roc_auc-old_auc:+.4f})")
    print(f"  最佳阈值: {best_thresh:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
