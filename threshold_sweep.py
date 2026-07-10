"""
=============================================================================
摔倒检测系统 - 阈值扫描脚本（方向A）
加载已训练的最佳模型，遍历不同阈值，输出完整指标表
=============================================================================
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到路径
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# 从训练脚本导入必要组件
from models.enhanced_lstm import build_enhanced_model

# ============ 配置 ============
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "enhanced_run_20260508_124532" / "best_model.pth"
FEATURES_DIR = BASE_DIR / "features"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_checkpoint(checkpoint_path):
    """加载训练时保存的checkpoint"""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return ckpt


def reconstruct_model(ckpt):
    """根据checkpoint中的config重建模型"""
    config = ckpt['config']
    model = build_enhanced_model(
        input_size=config['input_size'],
        model_type=config['model_type'],
        hidden_size=config['hidden_size'],
        num_classes=config['num_classes'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def standardize_test(X_test, scaler_mean, scaler_std):
    """使用训练时的统计量标准化测试集"""
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = (X_test_2d - scaler_mean) / scaler_std
    return X_test_scaled.reshape(X_test.shape)


def run_inference(model, X_test, batch_size=32):
    """运行推理，返回Fall类的概率"""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = torch.FloatTensor(X_test[i:i+batch_size])
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)[:, 1].numpy()
            all_probs.extend(probs)
    return np.array(all_probs)


def sweep_thresholds(y_true, y_prob, thresholds):
    """扫描不同阈值，返回指标字典列表"""
    results = []
    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(int)
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        results.append({
            'threshold': thresh,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
        })
    return results


def print_results_table(results):
    """打印格式化的结果表格"""
    print("\n" + "=" * 75)
    print("阈值扫描结果")
    print("=" * 75)
    print(f"{'阈值':>8} | {'准确率':>8} | {'精确率':>8} | {'召回率':>8} | {'F1分数':>8}")
    print("-" * 75)

    best_f1 = 0
    best_f1_thresh = 0
    best_recall_thresh = None

    for r in results:
        print(f"{r['threshold']:8.2f} | {r['accuracy']:8.4f} | {r['precision']:8.4f} | {r['recall']:8.4f} | {r['f1']:8.4f}")
        if r['f1'] > best_f1:
            best_f1 = r['f1']
            best_f1_thresh = r['threshold']
        if r['recall'] >= 0.90 and (best_recall_thresh is None or r['threshold'] < best_recall_thresh):
            best_recall_thresh = r['threshold']

    print("=" * 75)
    print(f"\n★ 最高F1分数: {best_f1:.4f} (阈值={best_f1_thresh:.2f})")
    if best_recall_thresh is not None:
        # 找到该阈值对应的精确率
        rec_result = next(r for r in results if r['threshold'] == best_recall_thresh)
        print(f"★ 满足召回率≥90%的最低阈值: {best_recall_thresh:.2f}")
        print(f"  对应指标: 精确率={rec_result['precision']:.4f}, 召回率={rec_result['recall']:.4f}, F1={rec_result['f1']:.4f}")
    else:
        print("★ 未找到召回率≥90%的阈值")
        # 找召回率最高的
        max_rec_result = max(results, key=lambda x: x['recall'])
        print(f"  最高召回率: {max_rec_result['recall']:.4f} (阈值={max_rec_result['threshold']:.2f})")
        print(f"  对应指标: 精确率={max_rec_result['precision']:.4f}, F1={max_rec_result['f1']:.4f}")


def save_results(results, save_path):
    """保存结果到JSON"""
    import json
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {save_path}")


def main():
    print("\n" + "=" * 60)
    print("摔倒检测系统 - 阈值扫描 (方向A)")
    print("=" * 60)

    # === 加载checkpoint ===
    print(f"\n[1] 加载checkpoint: {CHECKPOINT_PATH}")
    ckpt = load_checkpoint(CHECKPOINT_PATH)
    print(f"  训练时最佳F1: {ckpt['val_f1']:.4f}")
    print(f"  训练时最佳阈值: {ckpt['threshold']:.2f}")
    config = ckpt['config']
    print(f"  模型配置: {config}")

    # === 重建模型 ===
    print("\n[2] 重建模型...")
    model = reconstruct_model(ckpt)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {param_count:,}")

    # === 加载测试集 ===
    print("\n[3] 加载测试集...")
    test_X = np.load(FEATURES_DIR / "test" / "X.npy")
    test_y = np.load(FEATURES_DIR / "test" / "y.npy")
    print(f"  测试集: {test_X.shape}, 分布: Normal={np.sum(test_y==0)}, Fall={np.sum(test_y==1)}")

    # === 标准化 ===
    print("\n[4] 标准化测试集（使用训练集统计量）...")
    scaler_mean = ckpt['scaler_mean']
    scaler_std = ckpt['scaler_std']
    test_X_scaled = standardize_test(test_X, scaler_mean, scaler_std)

    # === 推理 ===
    print("\n[5] 运行推理...")
    y_prob = run_inference(model, test_X_scaled)
    auc = roc_auc_score(test_y, y_prob)
    print(f"  AUC-ROC: {auc:.4f}")

    # === 扫描阈值 ===
    print("\n[6] 扫描阈值 (0.05 ~ 0.95)...")
    thresholds = np.arange(0.05, 1.00, 0.05)
    results = sweep_thresholds(test_y, y_prob, thresholds)

    # === 打印结果 ===
    print_results_table(results)

    # === 保存结果 ===
    timestamp = ckpt.get('timestamp', '20260508_124532')
    save_path = RESULTS_DIR / f"threshold_sweep_{timestamp}.json"
    save_results(results, save_path)

    # === 建议 ===
    print("\n" + "=" * 60)
    print("建议:")
    best_f1_result = max(results, key=lambda x: x['f1'])
    recall_90_result = next((r for r in results if r['recall'] >= 0.90), None)
    if recall_90_result:
        print(f"  1. 若要满足召回率≥90%: 使用阈值 {recall_90_result['threshold']:.2f}")
        print(f"     精确率={recall_90_result['precision']:.4f}, 召回率={recall_90_result['recall']:.4f}, F1={recall_90_result['f1']:.4f}")
    print(f"  2. 若要最大化F1: 使用阈值 {best_f1_result['threshold']:.2f}")
    print(f"     精确率={best_f1_result['precision']:.4f}, 召回率={best_f1_result['recall']:.4f}, F1={best_f1_result['f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
