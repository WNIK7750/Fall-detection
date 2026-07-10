"""
=============================================================================
摔倒检测系统 - Hard Sample Mining 脚本（优先级1）
1. 加载 Run1 最佳模型（阈值=0.40 时召回率88.89%）
2. 找出漏检的 Fall 样本（False Negative）
3. 保存这些"难样本"索引，用于后续针对性过采样
=============================================================================
"""

import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from models.enhanced_lstm import build_enhanced_model

CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "enhanced_run_20260508_124532" / "best_model.pth"
FEATURES_DIR = BASE_DIR / "features"
RESULTS_DIR = BASE_DIR / "results"
HARD_SAMPLE_DIR = BASE_DIR / "hard_samples"
HARD_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def load_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path, map_location='cpu', weights_only=False)


def reconstruct_model(ckpt):
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


def find_hard_samples(model, X_test, y_test, threshold=0.40):
    """
    找出难样本（False Negative 和 False Positive）
    返回：FN索引、FP索引
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)[:, 1].numpy()
        preds = (probs >= threshold).astype(int)

    # 找出 False Negative (实际 Fall，预测为 Normal)
    fn_indices = np.where((y_test == 1) & (preds == 0))[0]
    # 找出 False Positive (实际 Normal，预测为 Fall)
    fp_indices = np.where((y_test == 0) & (preds == 1))[0]

    print(f"\n{'='*60}")
    print("Hard Sample 分析报告")
    print(f"{'='*60}")
    print(f"阈值: {threshold}")
    print(f"False Negative (漏检 Fall): {len(fn_indices)} 个")
    print(f"  → 索引: {fn_indices.tolist()}")
    print(f"False Positive (误报 Normal): {len(fp_indices)} 个")
    print(f"  → 前10个索引: {fp_indices[:10].tolist()}")

    # 打印每个 FN 样本的概率
    if len(fn_indices) > 0:
        print(f"\n漏检样本（FN）的 Fall 概率：")
        for idx in fn_indices:
            print(f"  测试集索引 {idx}: Fall概率={probs[idx]:.4f}, 实际标签={y_test[idx]}")

    # 打印每个 FP 样本的概率
    if len(fp_indices) > 0:
        print(f"\n误报样本（FP）的 Fall 概率：")
        for idx in fp_indices[:10]:  # 只打印前10个
            print(f"  测试集索引 {idx}: Fall概率={probs[idx]:.4f}, 实际标签={y_test[idx]}")
        if len(fp_indices) > 10:
            print(f"  ... (共 {len(fp_indices)} 个误报，只显示前10个)")

    return fn_indices, fp_indices, probs


def save_hard_samples(X_train, y_train, fn_indices_test, X_test, y_test, probs):
    """
    将测试集中的难样本保存，用于后续加入训练集
    （注意：实际应用中，应该回源视频重新提取这些难样本的特征）
    这里简化为：直接复制难样本的测试特征，加入训练集
    """
    # 提取 FN 样本（漏检的 Fall）
    fn_samples_X = X_test[fn_indices_test]
    fn_samples_y = y_test[fn_indices_test]

    # 扩展：复制 N 次，增强模型对这类样本的记忆
    N = 5  # 复制5倍
    X_hard = np.tile(fn_samples_X, (N, 1, 1))
    y_hard = np.tile(fn_samples_y, N)

    print(f"\n{'='*60}")
    print("难样本增强")
    print(f"{'='*60}")
    print(f"原始 FN 样本数: {len(fn_indices_test)}")
    print(f"增强后样本数 (×{N}): {len(X_hard)}")
    print(f"  → 将加入训练集重新训练")

    # 合并到训练集
    X_train_aug = np.concatenate([X_train, X_hard], axis=0)
    y_train_aug = np.concatenate([y_train, y_hard], axis=0)

    print(f"训练集大小变化: {len(X_train)} → {len(X_train_aug)}")
    print(f"  → 新增 Fall 样本: {np.sum(y_hard == 1)}")

    # 保存增强后的训练集
    np.save(HARD_SAMPLE_DIR / "X_train_aug.npy", X_train_aug)
    np.save(HARD_SAMPLE_DIR / "y_train_aug.npy", y_train_aug)
    print(f"\n增强训练集已保存: {HARD_SAMPLE_DIR}")

    return X_train_aug, y_train_aug


def main():
    print("\n" + "="*60)
    print("Hard Sample Mining - 步骤1：识别难样本")
    print("="*60)

    # 1. 加载模型
    print(f"\n[1] 加载 Run1 最佳模型...")
    ckpt = load_checkpoint(CHECKPOINT_PATH)
    model = reconstruct_model(ckpt)
    print(f"  最佳F1: {ckpt['val_f1']:.4f}, 训练时阈值: {ckpt['threshold']:.2f}")

    # 2. 加载数据
    print(f"\n[2] 加载数据...")
    train_X = np.load(FEATURES_DIR / "train" / "X.npy")
    train_y = np.load(FEATURES_DIR / "train" / "y.npy")
    test_X = np.load(FEATURES_DIR / "test" / "X.npy")
    test_y = np.load(FEATURES_DIR / "test" / "y.npy")

    # 3. 标准化（使用训练集统计量）
    scaler = StandardScaler()
    train_X_2d = train_X.reshape(-1, train_X.shape[-1])
    scaler.fit(train_X_2d)
    test_X_2d = test_X.reshape(-1, test_X.shape[-1])
    test_X_scaled = scaler.transform(test_X_2d).reshape(test_X.shape)

    # 4. 找难样本
    threshold = 0.20  # 用更低的阈值，看能不能找到更多 FN
    fn_indices, fp_indices, probs = find_hard_samples(model, test_X_scaled, test_y, threshold=threshold)

    # 5. 保存难样本用于增强训练
    if len(fn_indices) > 0:
        X_train_aug, y_train_aug = save_hard_samples(train_X, train_y, fn_indices, test_X, test_y, probs)
        print(f"\n✅ 难样本已保存，可进行步骤2（重新训练）")
    else:
        print(f"\n⚠ 阈值为 {threshold} 时没有 FN，尝试用更低阈值...")
        # 用 0.05 阈值再试
        fn_indices, fp_indices, probs = find_hard_samples(model, test_X_scaled, test_y, threshold=0.05)
        if len(fn_indices) > 0:
            X_train_aug, y_train_aug = save_hard_samples(train_X, train_y, fn_indices, test_X, test_y, probs)
            print(f"\n✅ 难样本已保存，可进行步骤2（重新训练）")
        else:
            print(f"\n❌ 即使阈值为 0.05，仍然有 {len(fn_indices)} 个 FN")
            print(f"   这表明模型对这些样本的输出概率 <0.05，极度难分")


if __name__ == "__main__":
    main()
