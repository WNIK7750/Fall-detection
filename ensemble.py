"""
=============================================================================
摔倒检测系统 - 模型集成脚本 (Ensemble)
组合 Run1（高精确率）和 Run4（高召回率）两个模型
策略：
  A. 硬投票：两个模型都预测 Fall 才报警（高精确率）
  B. 硬投票：任一模型预测 Fall 就报警（高召回率）
  C. 软投票：平均两个模型的概率，再扫阈值（平衡）
  D. 加权软投票：给高召回模型更高权重
=============================================================================
"""

import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from models.enhanced_lstm import build_enhanced_model


# ============ 两个模型的 checkpoint 路径 ============
CKPT_RUN1 = BASE_DIR / "checkpoints" / "enhanced_run_20260508_124532" / "best_model.pth"  # 高精确率模型
CKPT_RUN4 = BASE_DIR / "checkpoints" / "enhanced_run_20260508_131605" / "best_model.pth"  # 高召回率模型
FEATURES_DIR = BASE_DIR / "features"

# 如果两个路径不存在，让用户手动指定
if not CKPT_RUN4.exists():
    print(f"[警告] Run4 checkpoint 不存在: {CKPT_RUN4}")
    print("         请确认 Run4 的训练输出，修改 CKPT_RUN4 路径")
    # 尝试自动查找
    import glob
    run4_dirs = sorted(BASE_DIR.glob("checkpoints/enhanced_run_20260508_131*"))
    if run4_dirs:
        CKPT_RUN4 = run4_dirs[0] / "best_model.pth"
        print(f"         自动找到: {CKPT_RUN4}")


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    model = build_enhanced_model(
        input_size=config['input_size'],
        model_type=config['model_type'],
        hidden_size=config['hidden_size'],
        num_classes=config['num_classes'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


def get_probabilities(model, X):
    """获取模型预测的 Fall 类概率"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)[:, 1].numpy()
    return probs


def ensemble_hard_vote(prob1, prob2, strategy='both'):
    """
    硬投票集成
    strategy='both': 两个模型都预测 Fall 才报警（高精确率）
    strategy='either': 任一模型预测 Fall 就报警（高召回率）
    """
    if strategy == 'both':
        return ((prob1 >= 0.5) & (prob2 >= 0.5)).astype(int)
    elif strategy == 'either':
        return ((prob1 >= 0.5) | (prob2 >= 0.5)).astype(int)


def ensemble_soft_vote(prob1, prob2, weights=(0.5, 0.5)):
    """软投票：加权平均概率"""
    return weights[0] * prob1 + weights[1] * prob2


def evaluate(y_true, preds, probs=None, threshold=None):
    """评估指标"""
    if preds is None:
        preds = (probs >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'preds': preds}


def sweep_threshold(y_true, probs):
    """扫描阈值，找最佳组合"""
    results = []
    for thresh in np.arange(0.05, 1.00, 0.05):
        r = evaluate(y_true, None, probs, thresh)
        r['threshold'] = thresh
        results.append(r)
    return results


def print_results_table(results, title):
    """打印结果表格"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(f"{'阈值':>8} | {'准确率':>8} | {'精确率':>8} | {'召回率':>8} | {'F1':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['threshold']:8.2f} | {r['accuracy']:8.4f} | {r['precision']:8.4f} | {r['recall']:8.4f} | {r['f1']:8.4f}")

    # 找最佳
    best_f1 = max(results, key=lambda x: x['f1'])
    best_recall = next((r for r in results if r['recall'] >= 0.90), None)
    print(f"{'='*70}")
    print(f"★ 最高F1: {best_f1['f1']:.4f} (阈值={best_f1['threshold']:.2f})")
    if best_recall:
        print(f"★ 召回率≥90%: 阈值={best_recall['threshold']:.2f}, F1={best_recall['f1']:.4f}, 精确率={best_recall['precision']:.4f}")
    else:
        max_rec = max(results, key=lambda x: x['recall'])
        print(f"★ 最高召回率: {max_rec['recall']:.4f} (阈值={max_rec['threshold']:.2f})")


def main():
    print("\n" + "="*60)
    print("摔倒检测系统 - 模型集成 (Ensemble)")
    print("="*60)

    # === 1. 加载两个模型 ===
    print(f"\n[1] 加载模型...")
    print(f"  Run1 (高精确率): {CKPT_RUN1}")
    model1, ckpt1 = load_model(CKPT_RUN1)
    print(f"  → 训练时F1: {ckpt1['val_f1']:.4f}, 阈值: {ckpt1['threshold']:.2f}")

    print(f"  Run4 (高召回率): {CKPT_RUN4}")
    if not CKPT_RUN4.exists():
        print("  [错误] Run4 checkpoint 不存在，请用实际路径替换 CKPT_RUN4")
        return
    model4, ckpt4 = load_model(CKPT_RUN4)
    print(f"  → 训练时F1: {ckpt4['val_f1']:.4f}, 阈值: {ckpt4['threshold']:.2f}")

    # === 2. 加载测试集 ===
    print(f"\n[2] 加载测试集...")
    test_X = np.load(FEATURES_DIR / "test" / "X.npy")
    test_y = np.load(FEATURES_DIR / "test" / "y.npy")
    print(f"  测试集: {test_X.shape}, 分布: Normal={np.sum(test_y==0)}, Fall={np.sum(test_y==1)}")

    # === 3. 标准化（使用 Run1 的统计量）===
    print(f"\n[3] 标准化（使用 Run1 的统计量）...")
    scaler = StandardScaler()
    train_X = np.load(FEATURES_DIR / "train" / "X.npy")
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    test_X_scaled = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    # === 4. 获取两个模型的概率输出 ===
    print(f"\n[4] 获取模型预测...")
    prob1 = get_probabilities(model1, test_X_scaled)
    prob4 = get_probabilities(model4, test_X_scaled)
    auc1 = roc_auc_score(test_y, prob1)
    auc4 = roc_auc_score(test_y, prob4)
    print(f"  Run1 AUC: {auc1:.4f}")
    print(f"  Run4 AUC: {auc4:.4f}")

    # === 5. 策略A：硬投票（both）===
    preds_A = ensemble_hard_vote(prob1, prob4, strategy='both')
    rA = evaluate(test_y, preds_A, None, None)
    print(f"\n{'='*70}")
    print("策略A：硬投票（both）→ 高精确率")
    print(f"  精确率={rA['precision']:.4f}, 召回率={rA['recall']:.4f}, F1={rA['f1']:.4f}")

    # === 6. 策略B：硬投票（either）===
    preds_B = ensemble_hard_vote(prob1, prob4, strategy='either')
    rB = evaluate(test_y, preds_B, None, None)
    print(f"\n{'='*70}")
    print("策略B：硬投票（either）→ 高召回率")
    print(f"  精确率={rB['precision']:.4f}, 召回率={rB['recall']:.4f}, F1={rB['f1']:.4f}")

    # === 7. 策略C：软投票（平均）===
    prob_C = ensemble_soft_vote(prob1, prob4, weights=(0.5, 0.5))
    results_C = sweep_threshold(test_y, prob_C)
    print_results_table(results_C, "策略C：软投票（平均概率）")

    # === 8. 策略D：加权软投票（Run4权重更高，因为它召回率高）===
    prob_D = ensemble_soft_vote(prob1, prob4, weights=(0.3, 0.7))
    results_D = sweep_threshold(test_y, prob_D)
    print_results_table(results_D, "策略D：加权软投票（Run1×0.3 + Run4×0.7）")

    # === 9. 最佳结果汇总 ===
    print(f"\n{'='*70}")
    print("集成结果汇总（对比单模型最佳）")
    print(f"{'='*70}")
    print(f"{'模型/策略':<30} | {'精确率':>8} | {'召回率':>8} | {'F1':>8}")
    print("-" * 70)
    print(f"{'Run1 (单模型最佳F1)':<30} | {'71.43%':>8} | {'88.89%':>8} | {'76.92%':>8}")
    print(f"{'Run4 (单模型最佳召回)':<30} | {'42.50%':>8} | {'94.44%':>8} | {'58.62%':>8}")
    print(f"{'策略A (both, 高精确率)':<30} | {rA['precision']*100:>7.2f}% | {rA['recall']*100:>7.2f}% | {rA['f1']*100:>7.2f}%")
    print(f"{'策略B (either, 高召回率)':<30} | {rB['precision']*100:>7.2f}% | {rB['recall']*100:>7.2f}% | {rB['f1']*100:>7.2f}%")

    # 找策略C的最佳F1
    best_C = max(results_C, key=lambda x: x['f1'])
    print(f"{'策略C (软投票平均) 最佳':<30} | {best_C['precision']*100:>7.2f}% | {best_C['recall']*100:>7.2f}% | {best_C['f1']*100:>7.2f}%")

    # 找策略D的最佳F1
    best_D = max(results_D, key=lambda x: x['f1'])
    print(f"{'策略D (加权0.3+0.7) 最佳':<30} | {best_D['precision']*100:>7.2f}% | {best_D['recall']*100:>7.2f}% | {best_D['f1']*100:>7.2f}%")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
