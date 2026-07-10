# Demo01 - 基线训练阶段归档

## 归档时间
2026-05-12

## 内容说明

本目录包含 **demo01 阶段**（基线训练阶段）的完整训练过程和结果。

### 归档内容

| 目录/文件 | 说明 |
|-----------|------|
| `checkpoints/` | 所有训练checkpoint（143个.pth文件，8个训练run） |
| `features/` | 所有提取的特征数据（train/val/test的.npy文件） |
| `results/` | 所有训练和测试结果（JSON + PNG图表） |
| `models/enhanced_lstm.py` | 增强模型架构定义（BiLSTM+BiGRU+Attention） |
| `train_enhanced.py` | 增强训练脚本 |
| `ensemble.py` | 模型集成脚本 |
| `ensemble_inference.py` | 集成推理脚本 |
| `extract_features_v3.py` | 特征提取脚本 |
| `demo_video.py` | 演示视频推理脚本 |
| `hard_sample_mining.py` | 难例挖掘脚本 |
| `threshold_sweep.py` | 阈值扫描脚本 |
| `check_checkpoint.py` | checkpoint检查脚本 |
| `inspect_checkpoint.py` | checkpoint详情脚本 |
| `inspect_runs.py` | 训练run检查脚本 |
| `yolov8n-pose.pt` | YOLOv8姿态模型权重 |
| `执行指南.md` | 执行指南 |
| `README.md` | 项目说明 |
| `requirements.txt` | Python依赖 |

## 关键结果（Demo01阶段）

### 原始LSTM基线
- 架构：双层单向LSTM（64→32），无Attention
- 指标：Acc=73.1%, Prec=37.2%, Recall=88.9%, F1=52.5%, AUC=0.865

### 增强模型（BiLSTM+BiGRU+Attention, full）
- Run1（SMOTE+增强）：Acc=91.7%, Prec=71.4%, Recall=83.3%, F1=76.9%, AUC=0.946
- Run4（SMOTE，无增强）：Acc=77.8%, Prec=42.5%, Recall=94.4%, F1=58.6%, AUC=0.928

### 集成模型（Run1 + Run4 软投票）
- 指标：Acc=92.6%, Prec=75.0%, Recall=83.3%, F1=78.9%

## 使用的模型种类

详见根目录 `模型参数总结.md`。
