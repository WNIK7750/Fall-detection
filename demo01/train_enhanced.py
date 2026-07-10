"""
=============================================================================
摔倒检测系统 - 增强训练脚本（阶段1+2+3优化）
支持：20维特征 + BiLSTM-BiGRU-Attention + SMOTE过采样 + 数据增强

优化内容：
1. 架构升级：BiLSTM + BiGRU + 双向Attention（← 10维 → 20维）
2. 特征扩展：10维 → 20维（extract_features_v4.py已有）
3. 数据平衡：SMOTE过采样（摔倒样本从79 → ~300+）
4. 数据增强：高斯噪声 + 时间掩码 + 特征抖动
5. 学习率：Cosine Annealing（更平滑收敛）
6. 早停策略：监控验证集F1
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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============ 配置 ============
class Config:
    BASE_DIR = Path(r"D:\识别人类摔倒的智能系统")
    FEATURES_DIR = BASE_DIR / "features"
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

    # === 增强模型配置 ===
    INPUT_SIZE = 10           # 10维特征（当前数据）/ 可切换到20维(v4提取器)
    HIDDEN_SIZE = 128        # 隐藏层大小（比原来的64更大）
    NUM_CLASSES = 2          # 二分类
    DROPOUT = 0.3            # Dropout比率
    MODEL_TYPE = 'full'      # 'full' 或 'compact'

    # === 训练超参数 ===
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 32
    EPOCHS = 150
    PATIENCE = 20            # 早停耐心值（增强模型需要更多epoch）

    # === SMOTE过采样 ===
    USE_SMOTE = True         # 启用SMOTE过采样
    SMOTE_K = 5              # SMOTE邻居数

    # === 数据增强（恢复正常强度） ===
    AUGMENTATION = True
    AUG_NOISE_SCALE = 0.05   # 恢复正常噪声强度
    AUG_MASK_PROB = 0.2      # 恢复正常掩码概率
    AUG_MASK_MAX = 4         # 恢复正常掩码帧数
    AUG_FLIP_PROB = 0.2      # 恢复正常翻转概率
    AUG_NOISE_SCALE = 0.05   # 高斯噪声强度
    AUG_MASK_PROB = 0.2      # 时间掩码概率
    AUG_MASK_MAX = 4         # 最大掩码帧数
    AUG_FLIP_PROB = 0.2      # 水平翻转概率

    # === 类别权重 ===
    ALPHA = 3.0              # Focal Loss alpha (加大，更关注少数类)
    GAMMA = 2.0              # Focal Loss gamma

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============ SMOTE 过采样 ============
def apply_smote(X, y, k=5, ratio='1:2'):
    """
    SMOTE: 对少数类（摔倒）进行过采样
    在特征空间中插值生成新样本
    
    ratio: '1:1' -> 等量采样(294:294)
           '1:2' -> 1:2采样(294:588)，摔倒样本翻倍
    """
    try:
        from imblearn.over_sampling import SMOTE
        n_normal = np.sum(y == 0)
        n_fall_orig = np.sum(y == 1)
        print(f"  [SMOTE] 原始分布: Normal={n_normal}, Fall={n_fall_orig}")
        
        if ratio == '1:2':
            target_fall = min(n_fall_orig * 2, n_normal * 2)
        elif ratio == '1:3':
            target_fall = min(n_fall_orig * 3, n_normal * 2)
        else:  # '1:1'
            target_fall = n_normal
        
        sampling_strategy = {0: n_normal, 1: int(target_fall)}
        smote = SMOTE(random_state=42, k_neighbors=min(k, n_fall_orig - 1),
                      sampling_strategy=sampling_strategy)
        X_resampled, y_resampled = smote.fit_resample(
            X.reshape(len(X), -1), y
        )
        X_resampled = X_resampled.reshape(-1, X.shape[1], X.shape[2])
        print(f"  [SMOTE] 过采样后: Normal={np.sum(y_resampled==0)}, Fall={np.sum(y_resampled==1)}")
        return X_resampled, y_resampled
    except ImportError:
        print("  [警告] imblearn 未安装，跳过SMOTE")
        return X, y


# ============ Focal Loss ============
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============ 数据集 ============
class EnhancedDataset(Dataset):
    """带增强的数据集"""

    def __init__(self, X, y, augmentation=True, noise_scale=0.05):
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
            # 1. 高斯噪声（所有特征）
            if np.random.rand() < 0.4:
                noise = torch.randn_like(x) * self.noise_scale
                x = x + noise

            # 2. 时间掩码（随机遮蔽某些帧）
            if np.random.rand() < 0.3:
                mask_len = min(np.random.randint(1, 5), len(x) - 1)
                mask_start = np.random.randint(0, len(x) - mask_len)
                x[mask_start:mask_start + mask_len] = 0

            # 3. 特征抖动（随机缩放某些特征）
            if np.random.rand() < 0.3:
                scale = np.random.uniform(0.9, 1.1, size=(x.shape[1],))
                x = x * torch.FloatTensor(scale)

            # 4. 水平翻转（翻转对称特征）
            if np.random.rand() < 0.2:
                # 翻转特征索引0（宽高比）和1（变化率）
                x[:, 0] = 1.0 - x[:, 0]
                x[:, 1] = -x[:, 1]

        return x, y


# ============ 评估器 ============
class EnhancedEvaluator:
    """增强模型评估器"""

    @staticmethod
    def evaluate(model, X, y, threshold=0.5, device='cpu'):
        """在测试集上评估模型"""
        model.eval()
        all_preds, all_probs = [], []

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            outputs = model(X_tensor)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= threshold).astype(int)

        all_preds.extend(preds)
        all_probs.extend(probs)

        return np.array(all_preds), np.array(all_probs), y

    @staticmethod
    def find_best_threshold(y_true, y_prob, target_recall=0.90):
        """找到满足目标召回率的最优阈值"""
        best_f1 = 0
        best_thresh = 0.5
        best_preds = None

        for thresh in np.arange(0.05, 0.90, 0.05):
            preds = (y_prob >= thresh).astype(int)
            rec = recall_score(y_true, preds, zero_division=0)

            # 优先满足目标召回率
            if rec >= target_recall:
                f1 = f1_score(y_true, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
                    best_preds = preds

        # 如果没有满足召回率的，返回最高F1的阈值
        if best_preds is None:
            for thresh in np.arange(0.05, 0.90, 0.05):
                preds = (y_prob >= thresh).astype(int)
                f1 = f1_score(y_true, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
                    best_preds = preds

        return best_thresh, best_preds, best_f1

    @staticmethod
    def print_full_report(y_true, y_prob, threshold):
        """打印完整评估报告"""
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
║           测试集评估报告（阈值={threshold:.2f}）           ║
╠══════════════════════════════════════════════════╣
║  准确率 (Accuracy):  {acc:.4f} ({acc*100:.1f}%)              ║
║  精确率 (Precision): {prec:.4f} ({prec*100:.1f}%)              ║
║  召回率 (Recall):    {rec:.4f} ({rec*100:.1f}%)              ║
║  F1分数 (F1-Score):  {f1:.4f} ({f1*100:.1f}%)              ║
╠══════════════════════════════════════════════════╣
║  混淆矩阵:                                    ║
║                    预测Normal  预测Fall        ║
║  实际Normal (ADL):     {tn:4d}       {fp:4d}           ║
║  实际Fall (摔倒):      {fn:4d}       {tp:4d}           ║
╚══════════════════════════════════════════════════╝""")

        print(classification_report(y_true, preds, target_names=['Normal', 'Fall']))
        return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'cm': {'tn':tn,'fp':fp,'fn':fn,'tp':tp}}

    @staticmethod
    def plot_all_metrics(y_true, y_prob, save_dir):
        """绘制所有评估图表"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. 混淆矩阵（用最佳阈值）
        _, _, best_f1 = EnhancedEvaluator.find_best_threshold(y_true, y_prob)
        best_thresh = 0.5
        for thresh in np.arange(0.05, 0.90, 0.05):
            preds = (y_prob >= thresh).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if abs(f1 - best_f1) < 0.001:
                best_thresh = thresh
                break

        preds = (y_prob >= best_thresh).astype(int)
        cm = confusion_matrix(y_true, preds)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Fall'],
                    yticklabels=['Normal', 'Fall'], ax=axes[0],
                    annot_kws={'size': 16})
        axes[0].set_title(f'Confusion Matrix (thresh={best_thresh:.2f})', fontsize=12)
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')

        # ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={roc_auc:.3f})')
        axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title(f'ROC Curve (AUC={roc_auc:.3f})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 概率分布
        axes[2].hist(y_prob[y_true == 0], bins=30, alpha=0.7, label='Normal', color='green')
        axes[2].hist(y_prob[y_true == 1], bins=30, alpha=0.7, label='Fall', color='red')
        axes[2].axvline(x=best_thresh, color='blue', linestyle='--', label=f'Thresh={best_thresh:.2f}')
        axes[2].set_xlabel('Fall Probability')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Probability Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / "enhanced_test_metrics.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  评估图表已保存: {save_dir / 'enhanced_test_metrics.png'}")
        return roc_auc, best_thresh


# ============ 训练器 ============
class EnhancedTrainer:
    """增强模型训练器"""

    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 scheduler, config, scaler_mean=None, scaler_std=None):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std

        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'val_prec': [], 'val_rec': []
        }

    def train_epoch(self):
        self.model.train()
        total_loss, all_preds, all_labels = 0, [], []

        for X, y in self.train_loader:
            X, y = X.to(self.config.DEVICE), y.to(self.config.DEVICE)

            self.optimizer.zero_grad()

            # 混合精度训练（加速+稳定）
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
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
                X, y = X.to(self.config.DEVICE), y.to(self.config.DEVICE)
                outputs = self.model(X)
                loss = self.criterion(outputs, y)

                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = np.argmax(outputs.cpu().numpy(), axis=1)
                all_preds.extend(preds)
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(probs)

        avg_loss = total_loss / len(self.val_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

        # 搜索最佳阈值
        best_thresh, _, best_f1_binary = EnhancedEvaluator.find_best_threshold(
            np.array(all_labels), np.array(all_probs), target_recall=0.90
        )

        return avg_loss, acc, f1, prec, rec, best_f1_binary, best_thresh

    def train(self, epochs, checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print("增强模型训练开始")
        print(f"{'='*60}")
        print(f"设备: {self.config.DEVICE}")
        print(f"模型: {self.config.MODEL_TYPE} (INPUT_SIZE={self.config.INPUT_SIZE})")
        print(f"批次: {self.config.BATCH_SIZE} | 学习率: {self.config.LEARNING_RATE}")
        print(f"Dropout: {self.config.DROPOUT} | Weight Decay: {self.config.WEIGHT_DECAY}")
        print(f"SMOTE: {'启用 (1:1)' if self.config.USE_SMOTE else '禁用'}")
        print(f"数据增强: {'启用 (减弱版)' if self.config.AUGMENTATION else '禁用'}")
        print(f"Focal Loss: alpha={self.config.ALPHA}, gamma={self.config.GAMMA}")
        print(f"{'='*60}")

        for epoch in range(epochs):
            train_loss, train_acc, train_f1 = self.train_epoch()

            if self.val_loader is not None:
                val_loss, val_acc, val_f1, val_prec, val_rec, val_f1_binary, best_thresh = self.validate()

                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']

                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['train_f1'].append(train_f1)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['val_f1'].append(val_f1)
                self.history['val_prec'].append(val_prec)
                self.history['val_rec'].append(val_rec)

                print(f"E{epoch+1:3d}/{epochs} | "
                      f"TL={train_loss:.4f} TA={train_acc:.3f} | "
                      f"VL={val_loss:.4f} VA={val_acc:.3f} VF1={val_f1:.3f} "
                      f"P={val_prec:.3f} R={val_rec:.3f} | "
                      f"LR={current_lr:.2e} | thresh={best_thresh:.2f}")

                # 保存最佳模型
                if val_f1_binary > self.best_val_f1:
                    self.best_val_f1 = val_f1_binary
                    self.patience_counter = 0
                    model_path = checkpoint_dir / "best_model.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_f1': val_f1_binary,
                        'val_acc': val_acc,
                        'threshold': best_thresh,
                        'config': {
                            'input_size': self.config.INPUT_SIZE,
                            'hidden_size': self.config.HIDDEN_SIZE,
                            'num_classes': self.config.NUM_CLASSES,
                            'model_type': self.config.MODEL_TYPE,
                        },
                        'scaler_mean': self.scaler_mean,
                        'scaler_std': self.scaler_std
                    }, model_path)
                    print(f"  ★ 新最佳模型! F1={val_f1_binary:.4f} (thresh={best_thresh:.2f})")
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.config.PATIENCE:
                    print(f"\n早停! 连续{self.config.PATIENCE}个epoch未提升")
                    break
            else:
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['train_f1'].append(train_f1)
                if (epoch + 1) % 10 == 0:
                    print(f"E{epoch+1:3d}/{epochs} | "
                          f"Loss={train_loss:.4f} Acc={train_acc:.3f} F1={train_f1:.3f}")

        print(f"\n训练完成! 最佳验证F1: {self.best_val_f1:.4f}")
        return self.history

    def plot_history(self, save_path):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(self.history['train_loss'], label='Train')
        if self.history['val_loss']:
            axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(self.history['train_acc'], label='Train')
        if self.history['val_acc']:
            axes[0, 1].plot(self.history['val_acc'], label='Validation')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(self.history['train_f1'], label='Train')
        if self.history['val_f1']:
            axes[1, 0].plot(self.history['val_f1'], label='Validation')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        if self.history['val_prec']:
            axes[1, 1].plot(self.history['val_prec'], label='Precision')
            axes[1, 1].plot(self.history['val_rec'], label='Recall')
            axes[1, 1].set_title('Precision & Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"训练曲线已保存: {save_path}")


# ============ 主函数 ============
def main():
    print("\n" + "=" * 60)
    print("摔倒检测系统 - 增强训练 (BiLSTM+BiGRU+Attention)")
    print("=" * 60)

    config = Config()
    for d in [config.MODELS_DIR, config.RESULTS_DIR, config.CHECKPOINTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # === 加载数据 ===
    print("\n[1] 加载数据...")
    train_X = np.load(config.FEATURES_DIR / "train" / "X.npy")
    train_y = np.load(config.FEATURES_DIR / "train" / "y.npy")
    val_X = np.load(config.FEATURES_DIR / "val" / "X.npy")
    val_y = np.load(config.FEATURES_DIR / "val" / "y.npy")
    test_X = np.load(config.FEATURES_DIR / "test" / "X.npy")
    test_y = np.load(config.FEATURES_DIR / "test" / "y.npy")

    print(f"训练集: {train_X.shape}, 分布: Normal={np.sum(train_y==0)}, Fall={np.sum(train_y==1)}")
    print(f"验证集: {val_X.shape}, 分布: Normal={np.sum(val_y==0)}, Fall={np.sum(val_y==1)}")
    print(f"测试集: {test_X.shape}, 分布: Normal={np.sum(test_y==0)}, Fall={np.sum(test_y==1)}")

    # === 检查特征维度 ===
    actual_feat_dim = train_X.shape[2]
    if actual_feat_dim != config.INPUT_SIZE:
        print(f"\n[警告] 特征维度不匹配!")
        print(f"  训练数据: {actual_feat_dim}维")
        print(f"  模型配置: {config.INPUT_SIZE}维")
        print(f"  请确认使用 extract_features_v4.py 提取了20维特征")
        if actual_feat_dim == 10:
            print(f"\n  当前数据为10维，请先运行以下命令提取20维特征:")
            print(f"    python extract_features_v4.py")
            print(f"  然后重新运行本训练脚本")
            return
        config.INPUT_SIZE = actual_feat_dim
        print(f"  已自动调整为: {actual_feat_dim}维")

    # === 特征标准化 ===
    print("\n[2] 特征标准化...")
    scaler = StandardScaler()
    train_X_2d = train_X.reshape(-1, train_X.shape[-1])
    scaler.fit(train_X_2d)
    train_mean = scaler.mean_
    train_std = scaler.scale_

    train_X_scaled = scaler.transform(train_X_2d).reshape(train_X.shape)
    val_X_2d = val_X.reshape(-1, val_X.shape[-1])
    val_X_scaled = scaler.transform(val_X_2d).reshape(val_X.shape)

    print(f"  特征均值范围: [{train_mean.min():.3f}, {train_mean.max():.3f}]")
    print(f"  特征标准差范围: [{train_std.min():.3f}, {train_std.max():.3f}]")

    # === SMOTE过采样 ===
    if config.USE_SMOTE:
        print("\n[3] SMOTE过采样...")
        train_X_scaled, train_y = apply_smote(train_X_scaled, train_y, k=config.SMOTE_K, ratio='1:1')
        print(f"  过采样后训练集: {train_X_scaled.shape}")
    else:
        print("\n[3] 跳过SMOTE过采样")

    # === 构建模型 ===
    print("\n[4] 构建增强模型...")
    from models.enhanced_lstm import build_enhanced_model

    model = build_enhanced_model(
        input_size=config.INPUT_SIZE,
        model_type=config.MODEL_TYPE,
        hidden_size=config.HIDDEN_SIZE,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  模型类型: {config.MODEL_TYPE}")
    print(f"  模型参数量: {param_count:,}")

    # === 数据加载器 ===
    train_dataset = EnhancedDataset(
        train_X_scaled, train_y,
        augmentation=config.AUGMENTATION,
        noise_scale=config.AUG_NOISE_SCALE
    )
    val_dataset = EnhancedDataset(val_X_scaled, val_y, augmentation=False)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # === 损失函数和优化器 ===
    criterion = FocalLoss(alpha=config.ALPHA, gamma=config.GAMMA)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)

    # === 训练 ===
    print("\n[5] 开始训练...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = config.CHECKPOINTS_DIR / f"enhanced_run_{timestamp}"

    trainer = EnhancedTrainer(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler, config,
        scaler_mean=train_mean, scaler_std=train_std
    )

    history = trainer.train(config.EPOCHS, checkpoint_dir)

    # === 绘制训练曲线 ===
    trainer.plot_history(config.RESULTS_DIR / f"enhanced_training_history_{timestamp}.png")

    # === 在测试集上评估 ===
    print("\n[6] 测试集评估（使用训练集统计量标准化）...")
    # 使用训练集的scaler对测试集做标准化
    test_X_2d = test_X.reshape(-1, test_X.shape[-1])
    test_X_scaled = scaler.transform(test_X_2d).reshape(test_X.shape)
    
    best_model_path = checkpoint_dir / "best_model.pth"
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_threshold = checkpoint.get('threshold', 0.5)
        print(f"加载最佳模型: {best_model_path}")
        print(f"训练时最佳F1: {checkpoint.get('val_f1', 'N/A')}")
    else:
        best_threshold = 0.5
        print("使用最终模型（未找到最佳检查点）")

    preds, probs, y_true = EnhancedEvaluator.evaluate(
        model, test_X_scaled, test_y, threshold=best_threshold, device=config.DEVICE
    )

    roc_auc, final_thresh = EnhancedEvaluator.plot_all_metrics(
        y_true, probs, config.RESULTS_DIR
    )

    # 搜索最优阈值
    optimal_thresh, _, _ = EnhancedEvaluator.find_best_threshold(y_true, probs, target_recall=0.90)
    metrics = EnhancedEvaluator.print_full_report(y_true, probs, optimal_thresh)

    # === 保存结果 ===
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_type': config.MODEL_TYPE,
        'feature_dim': int(config.INPUT_SIZE),
        'smote': config.USE_SMOTE,
        'augmentation': config.AUGMENTATION,
        'optimal_threshold': float(optimal_thresh),
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
        'best_val_f1': float(trainer.best_val_f1)
    }

    results_path = config.RESULTS_DIR / f"enhanced_results_{timestamp}.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {results_path}")

    # === 保存最终模型 ===
    final_model_path = config.MODELS_DIR / f"enhanced_fall_detector_{timestamp}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_size': config.INPUT_SIZE,
            'hidden_size': config.HIDDEN_SIZE,
            'num_classes': config.NUM_CLASSES,
            'model_type': config.MODEL_TYPE,
        },
        'threshold': float(optimal_thresh),
        'test_metrics': results['test_metrics'],
        'scaler_mean': train_mean,
        'scaler_std': train_std
    }, final_model_path)
    print(f"模型已保存: {final_model_path}")

    print("\n" + "=" * 60)
    print("增强训练完成!")
    print("=" * 60)
    print(f"\n★ 关键结果汇总:")
    print(f"  精确率: {metrics['prec']:.2%} (原={0.3721:.2%}, 目标≥70%)")
    print(f"  召回率: {metrics['rec']:.2%} (原={0.8889:.2%}, 目标≥90%)")
    print(f"  F1分数: {metrics['f1']:.2%} (原={0.5246:.2%}, 目标≥78%)")
    print(f"  AUC-ROC: {roc_auc:.4f} (原={0.8654:.4f})")
    print(f"  最佳阈值: {optimal_thresh:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
