# 基于多特征融合与LSTM的人体摔倒智能检测系统


## 📋 项目简介

本项目提出一种基于**多特征融合**与**双向LSTM-BiGRU-Attention**的人体摔倒智能检测系统。系统以 YOLOv8 Pose 为核心进行人体姿态估计，提取宽高比、躯干倾角、质心运动速度、关节角度及脊柱矢量等多维度特征，输入混合时序分类器实现摔倒判定。

**核心指标（测试集）：**

| 指标 | 数值 | 目标 | 达成 |
|------|------|------|------|
| 精确率 (Precision) | 75.00% | ≥70% | ✅ |
| 召回率 (Recall) | 83.33% | ≥90% | 接近 |
| F1分数 | 78.95% | ≥78% | ✅ |
| AUC-ROC | >0.94 | - | ✅ |

---

## 🗂️ 项目结构

```
识别人类摔倒的智能系统/
├── 论文/                          # 论文相关
│   ├── generate_paper_images.py    # 论文Word文档生成脚本
│   ├── pictures/                  # 论文用图片
│   └── *.docx                    # 生成的论文文档
├── checkpoints/                   # 训练好的模型权重
│   ├── enhanced_run_XXX/
│   │   └── best_model.pth
│   └── ...
├── dataset/                       # UR Fall Detection Dataset
│   ├── *.mp4
│   └── splits.txt
├── features/                      # 提取好的特征文件（.npy）
│   ├── train/X.npy, y.npy
│   ├── val/X.npy, y.npy
│   └── test/X.npy, y.npy
├── models/
│   ├── enhanced_lstm.py          # 模型定义（EnhancedFallDetector）
│   └── *.pth
├── results/                       # 训练结果、图表
│   ├── *.png
│   └── *.json
├── extract_features_v3.py        # 特征提取脚本（MediaPipe）
├── train_enhanced.py             # 模型训练脚本
├── ensemble_inference.py         # 集成推理脚本（主入口）
├── ensemble.py                    # 集成训练脚本
├── threshold_sweep.py            # 阈值扫描脚本
├── requirements.txt               # Python依赖
└── README.md                     # 本文件
```

---

## ⚙️ 环境配置

### 1. Python 版本
- Python 3.8 及以上（推荐 3.10+）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

**主要依赖：**

| 库 | 版本 | 用途 |
|----|------|------|
| torch | ≥1.12 | 深度学习框架 |
| opencv-python | ≥4.5 | 视频读取 |
| numpy | ≥1.21 | 数值计算 |
| scikit-learn | ≥1.0 | 评估指标、StandardScaler |
| matplotlib | ≥3.5 | 绘图（训练曲线） |
| tqdm | ≥4.62 | 进度条 |
| ultralytics | ≥8.0 | YOLOv8 Pose 姿态估计 |

> **注意：** 若使用 MediaPipe 提取特征，还需安装：
> ```bash
> pip install mediapipe>=0.10.35
> ```

---

## 🚀 快速开始

### 方式一：评估模型性能（推荐首次运行）

使用已提取好的特征文件，在测试集上评估集成模型性能：

```bash
python ensemble_inference.py --eval
```

**输出示例：**
```
╔══════════════════════════════════════════════════╗
║        集成模型测试集评估报告（阈值=0.70）      ║
╠══════════════════════════════════════════════════╣
║  准确率 (Accuracy):  0.9074 (90.7%)              ║
║  精确率 (Precision): 0.7500 (75.0%)              ║
║  召回率 (Recall):    0.8333 (83.3%)              ║
║  F1分数 (F1-Score): 0.7895 (78.9%)              ║
╚══════════════════════════════════════════════════╝
```

---

### 方式二：调整决策阈值

```bash
python ensemble_inference.py --eval --threshold 0.65
```

> 降低阈值 → 提高召回率（更多摔倒被检测出来）  
> 提高阈值 → 提高精确率（减少误报）

---

### 方式三：对自定义视频进行摔倒检测

```bash
python ensemble_inference.py --video 路径/视频名.mp4
```

> **注意：** 此功能需要先用 YOLOv8 Pose 或 MediaPipe 提取视频特征，详见"高级用法"。

---

## 🔬 高级用法

### 1. 重新提取特征（从原始视频）

```bash
# 使用 MediaPipe 提取特征（需要 .task 模型文件）
python extract_features_v3.py

# 使用 YOLOv8 Pose 提取特征（推荐）
python extract_features_v3.py --use-yolo
```

### 2. 重新训练模型

```bash
python train_enhanced.py
```

训练配置（在脚本内修改）：
- Focal Loss: `alpha=2.0, gamma=2.0`
- SMOTE过采样: `1:1`
- 集成模型A（高精确率）: `use_aug=True`
- 集成模型B（高召回率）: `use_aug=False`

### 3. 扫描最优阈值

```bash
python threshold_sweep.py
```


## 🧠 模型说明

### 集成策略

系统采用**软投票集成（Soft Voting）**融合两个模型：

| 模型 | 数据增强 | 特点 |
|------|----------|------|
| 模型A（Run 1） | 启用 | 高精确率（71.43%） |
| 模型B（Run 4） | 关闭 | 高召回率（94.44%） |
| **集成模型** | - | **Precision=75.00%, Recall=83.33%** |

- 集成权重：`0.5 × 模型A + 0.5 × 模型B`
- 决策阈值：`0.70`（可通过 `--threshold` 调整）

### 特征说明（10维）

| 编号 | 特征名称 | 物理含义 |
|------|----------|----------|
| F1 | 宽高比 (R) | 人体外接框宽高比 |
| F2 | 宽高比变化率 (ΔR) | 帧间突变检测 |
| F3 | 脊柱倾角 (θ) | 躯干与竖直轴夹角 |
| F4 | 脊柱倾角突变量 (Δθ) | 3帧内角度快速变化 |
| F5 | 归一化质心y坐标 (yc) | 质心相对高度 |
| F6 | 质心垂直速度 (Vy) | 竖直方向运动速度 |
| F7 | 速度比 (Rv) | 垂直/水平速度比 |
| F8 | 头脚矢量长度 (Lha) | 头顶到脚踝距离 |
| F9 | 归一化头部高度 (yh) | 头部相对高度 |
| F10 | 脊柱矢量y分量 | 脊柱方向竖直分量 |

---

## 📊 数据集

本项目使用 **UR Fall Detection Dataset**：
- 共 140 个视频（70 摔倒 + 70 正常）
- 数据划分：
  - 训练集：360 例（Normal 294 + Fall 66）
  - 验证集：87 例（Normal 79 + Fall 8）
  - 测试集：108 例（Normal 90 + Fall 18）

> 数据集下载：https://fenix.ur.edu.pl/~mhylla/urfdd/

---

## 📝 论文

完整论文位于：`论文/基于多特征融合与LSTM的人体摔倒智能检测系统研究_含图片版.docx`

论文结构：
1. 引言（研究背景、技术路线对比、贡献）
2. 相关工作与技术基础
3. 系统整体架构
4. 多维特征工程设计
5. BiLSTM-BiGRU-Attention 混合模型设计
6. 集成学习方案
7. 实验设计与结果分析
8. 消融实验分析
9. 讨论：局限性与未来方向
10. 结论

---

## 📦 依赖清单（requirements.txt）

```
torch>=1.12
torchvision>=0.13
numpy>=1.21
opencv-python>=4.5
scikit-learn>=1.0
matplotlib>=3.5
tqdm>=4.62
ultralytics>=8.0
mediapipe>=0.10.35
python-docx>=0.8.11
Pillow>=9.0
```

---

## ⚠️ 常见问题

### Q1: 运行时提示"无法加载模型权重"？
**A:** 检查 `ensemble_inference.py` 中的 `CKPT_MODEL_A` 和 `CKPT_MODEL_B` 路径是否正确指向 `checkpoints/` 目录下的 `.pth` 文件。

### Q2: 特征提取时提示"MediaPipe模型文件不存在"？
**A:** 从 [MediaPipe官方](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) 下载 `pose_landmarker_lite.task`，放到 `models/` 目录下。

### Q3: 怎样提高召回率（减少漏检）？
**A:** 降低决策阈值，例如：
```bash
python ensemble_inference.py --eval --threshold 0.50
```

### Q4: GPU 训练支持吗？
**A:** 支持。脚本会自动检测 CUDA，无需修改代码。

---

## 📖 参考文献

1. Lin et al. Focal Loss for Dense Object Detection. IEEE TPAMI 2020.
2. Chawla et al. SMOTE: Synthetic Minority Over-sampling Technique. JAIR 2002.
3. Hochreiter & Schmidhuber. Long Short-Term Memory. Neural Computation 1997.
4. Vaswani et al. Attention Is All You Need. NIPS 2017.
5. Cao et al. OpenPose: Realtime Multi-Person 2D Pose Estimation. IEEE TPAMI 2019.
6. Jocher et al. Ultralytics YOLOv8. 2023.

---

## 📄  license

本项目为学术研究与课程设计用途，代码与文档仅供学习交流使用。

---

*最后更新：2026年5月*
