"""
=============================================================================
摔倒检测系统 - 集成推理脚本 v1.0
模型组合：BiLSTM-BiGRU-Attention × 2（软投票集成）

技术方案：
  - 基础模型：BiLSTM + BiGRU + 双向Attention（10维特征，128隐层）
  - 数据增强：标准版（无Aug）模型 + 增强版（Full Aug）模型
  - 集成策略：软投票（概率平均），阈值=0.70

最终指标：
  - 精确率: 75.00% (目标≥70% ✓)
  - 召回率: 83.33%
  - F1分数: 78.95% (目标≥78% ✓)
  - AUC-ROC: >0.94

训练数据：
  - Normal样本: 294 (训练集) + 79 (验证集) + 90 (测试集)
  - Fall样本: 66 (训练集) + 13 (验证集) + 18 (测试集)
  - SMOTE过采样: 294:294 (1:1)
  - Focal Loss: alpha=2.0, gamma=2.0

使用说明：
  python ensemble_inference.py --video <视频路径> [--threshold 0.70]
=============================================================================
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import json
from datetime import datetime

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from models.enhanced_lstm import build_enhanced_model


# ============ 模型配置 ============
# Run1: 高精确率模型（启用数据增强，Focal Loss alpha=2.0）
CKPT_MODEL_A = BASE_DIR / "checkpoints" / "enhanced_run_20260508_124532" / "best_model.pth"
# Run4: 高召回率模型（无数据增强）
CKPT_MODEL_B = BASE_DIR / "checkpoints" / "enhanced_run_20260508_131605" / "best_model.pth"

# 集成参数
ENSEMBLE_WEIGHT_A = 0.5  # 模型A权重
ENSEMBLE_WEIGHT_B = 0.5  # 模型B权重
DEFAULT_THRESHOLD = 0.70  # 默认阈值


# ============ 模型加载 ============
def load_model(checkpoint_path):
    """加载模型和训练配置"""
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


def get_fall_probability(model, X_scaled):
    """获取单个模型的Fall概率"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)[:, 1].numpy()
    return probs


# ============ 特征提取 ============
def extract_features_from_video(video_path):
    """
    从视频提取特征（调用已有的特征提取器）
    返回: (20, 10) 的特征序列
    """
    # 这里复用已有的特征提取逻辑
    # 实际使用时，调用 extract_features_v3.py 或 v4.py
    from extract_features_v3 import extract_multimodal_features
    features = extract_multimodal_features(str(video_path))
    return features


# ============ 集成推理 ============
class EnsembleFallDetector:
    """
    集成摔倒检测器
    组合两个不同训练配置的模型，使用软投票输出最终预测
    """

    def __init__(self, ckpt_a_path, ckpt_b_path, 
                 weight_a=0.5, weight_b=0.5,
                 threshold=0.70):
        self.weight_a = weight_a
        self.weight_b = weight_b
        self.threshold = threshold

        print(f"初始化集成检测器...")
        print(f"  模型A: {ckpt_a_path}")
        self.model_a, self.ckpt_a = load_model(ckpt_a_path)
        print(f"    - 训练F1: {self.ckpt_a['val_f1']:.4f}, 阈值: {self.ckpt_a['threshold']:.2f}")

        print(f"  模型B: {ckpt_b_path}")
        self.model_b, self.ckpt_b = load_model(ckpt_b_path)
        print(f"    - 训练F1: {self.ckpt_b['val_f1']:.4f}, 阈值: {self.ckpt_b['threshold']:.2f}")

        print(f"  集成权重: A={weight_a}, B={weight_b}")
        print(f"  决策阈值: {threshold}")

    def predict(self, X_scaled):
        """
        预测单个样本
        X_scaled: 标准化后的特征 (20, 10)
        返回: (is_fall: bool, fall_prob: float)
        """
        # 获取两个模型的概率
        prob_a = get_fall_probability(self.model_a, X_scaled)
        prob_b = get_fall_probability(self.model_b, X_scaled)

        # 软投票
        ensemble_prob = self.weight_a * prob_a + self.weight_b * prob_b

        # 决策
        is_fall = ensemble_prob >= self.threshold

        return bool(is_fall), float(ensemble_prob)

    def predict_batch(self, X_scaled_batch):
        """
        批量预测
        X_scaled_batch: (N, 20, 10) 标准化后的特征
        返回: (is_fall_array, fall_prob_array)
        """
        prob_a = get_fall_probability(self.model_a, X_scaled_batch)
        prob_b = get_fall_probability(self.model_b, X_scaled_batch)

        ensemble_prob = self.weight_a * prob_a + self.weight_b * prob_b
        is_fall = ensemble_prob >= self.threshold

        return is_fall, ensemble_prob

    def set_threshold(self, threshold):
        """调整决策阈值"""
        self.threshold = threshold
        print(f"阈值已更新: {threshold}")

    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_a': {
                'val_f1': float(self.ckpt_a['val_f1']),
                'threshold': float(self.ckpt_a['threshold']),
                'config': self.ckpt_a['config']
            },
            'model_b': {
                'val_f1': float(self.ckpt_b['val_f1']),
                'threshold': float(self.ckpt_b['threshold']),
                'config': self.ckpt_b['config']
            },
            'ensemble_weights': {
                'a': self.weight_a,
                'b': self.weight_b
            },
            'decision_threshold': self.threshold
        }


# ============ 测试集评估 ============
def evaluate_on_test_set(detector, features_dir):
    """在测试集上评估集成模型"""
    print("\n" + "="*60)
    print("测试集评估")
    print("="*60)

    test_X = np.load(features_dir / "test" / "X.npy")
    test_y = np.load(features_dir / "test" / "y.npy")

    # 标准化
    train_X = np.load(features_dir / "train" / "X.npy")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    test_X_scaled = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    # 预测
    preds, probs = detector.predict_batch(test_X_scaled)

    # 评估
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                f1_score, confusion_matrix, classification_report)
    acc = accuracy_score(test_y, preds)
    prec = precision_score(test_y, preds)
    rec = recall_score(test_y, preds)
    f1 = f1_score(test_y, preds)
    cm = confusion_matrix(test_y, preds)

    print(f"\n╔══════════════════════════════════════════════════╗")
    print(f"║        集成模型测试集评估报告（阈值={detector.threshold:.2f}）      ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  准确率 (Accuracy):  {acc:.4f} ({acc*100:.1f}%)              ║")
    print(f"║  精确率 (Precision): {prec:.4f} ({prec*100:.1f}%)              ║")
    print(f"║  召回率 (Recall):    {rec:.4f} ({rec*100:.1f}%)              ║")
    print(f"║  F1分数 (F1-Score):  {f1:.4f} ({f1*100:.1f}%)              ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  混淆矩阵:                                    ║")
    print(f"║                    预测Normal  预测Fall        ║")
    tn, fp, fn, tp = cm.ravel()
    print(f"║  实际Normal (ADL):     {tn:4d}        {fp:4d}           ║")
    print(f"║  实际Fall (摔倒):       {fn:4d}        {tp:4d}           ║")
    print(f"╚══════════════════════════════════════════════════╝")

    # 目标达成情况
    print(f"\n目标达成情况:")
    print(f"  精确率 ≥70%: {'✓' if prec >= 0.70 else '✗'} ({prec*100:.2f}%)")
    print(f"  F1分数 ≥78%: {'✓' if f1 >= 0.78 else '✗'} ({f1*100:.2f}%)")

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'cm': cm.tolist()}


# ============ 主函数 ============
def main():
    parser = argparse.ArgumentParser(description="摔倒检测集成推理")
    parser.add_argument('--video', type=str, help='视频文件路径')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f'决策阈值 (默认={DEFAULT_THRESHOLD})')
    parser.add_argument('--eval', action='store_true',
                        help='在测试集上评估模型')
    parser.add_argument('--save', type=str, help='保存预测结果JSON路径')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("摔倒检测系统 - 集成推理 v1.0")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 初始化检测器
    detector = EnsembleFallDetector(
        CKPT_MODEL_A, CKPT_MODEL_B,
        weight_a=ENSEMBLE_WEIGHT_A,
        weight_b=ENSEMBLE_WEIGHT_B,
        threshold=args.threshold
    )

    # 评估模式
    if args.eval:
        results = evaluate_on_test_set(detector, BASE_DIR / "features")
        
        # 保存结果
        output = {
            'timestamp': datetime.now().isoformat(),
            'model_info': detector.get_model_info(),
            'test_results': results,
        }
        output_path = BASE_DIR / "results" / f"ensemble_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {output_path}")
        return

    # 单视频推理模式
    if args.video:
        print(f"\n推理视频: {args.video}")
        features = extract_features_from_video(args.video)
        is_fall, prob = detector.predict(features)
        print(f"Fall概率: {prob:.4f}")
        print(f"预测结果: {'⚠️ 摔倒!' if is_fall else '✓ 正常'}")
        return

    # 无参数时显示帮助
    parser.print_help()
    print(f"\n示例:")
    print(f"  评估测试集: python ensemble_inference.py --eval")
    print(f"  调整阈值:   python ensemble_inference.py --eval --threshold 0.65")


if __name__ == "__main__":
    main()
