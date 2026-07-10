"""
=============================================================================
摔倒检测系统 - 演示脚本
功能：播放本地视频，实时显示姿态估计、特征可视化、摔倒检测

选项配置：
- 视频来源：本地视频文件（MP4）
- 标记内容：丰富模式（关键点 + BBox + 脊柱倾角 + 概率曲线）
- 状态标记：右上角文字显示
- 保存结果：否（仅显示）

依赖：opencv-python, numpy, torch, ultralytics
=============================================================================
"""

import sys
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from collections import deque

# 路径配置
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from models.enhanced_lstm import build_enhanced_model

# ============ 演示播放列表配置 ============
# 只演示前两个 + 后两个（共4个视频）
DEMO_PLAYLIST = [
    BASE_DIR / "dataset" / "test" / "fall" / "fall-07-cam0.mp4",
    BASE_DIR / "dataset" / "test" / "adl" / "adl-20-cam0.mp4",
    BASE_DIR / "dataset" / "test" / "fall" / "fall-25-cam0.mp4",
    BASE_DIR / "dataset" / "test" / "adl" / "adl-16-cam0.mp4",
]

# ============ 模型配置 ============
CKPT_MODEL_A = BASE_DIR / "checkpoints" / "enhanced_run_20260508_124532" / "best_model.pth"
CKPT_MODEL_B = BASE_DIR / "checkpoints" / "enhanced_run_20260508_131605" / "best_model.pth"
ENSEMBLE_WEIGHT_A = 0.5
ENSEMBLE_WEIGHT_B = 0.5
DEFAULT_THRESHOLD = 0.70
FEATURE_DIM = 10
WINDOW_SIZE = 20   # 必须与训练时一致！不要改动
WINDOW_STRIDE = 2  # 跳帧间隔（越小越早介入，但计算量增加）

# ============ YOLOv8 Pose 配置 ============
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("警告: ultralytics 未安装，将使用模拟姿态数据")


# ============ COCO 关键点定义 ============
COCO_KEYPOINTS = [
    'Nose', 'L.Eye', 'R.Eye', 'L.Ear', 'R.Ear',
    'L.Shoulder', 'R.Shoulder', 'L.Elbow', 'R.Elbow',
    'L.Wrist', 'R.Wrist', 'L.Hip', 'R.Hip',
    'L.Knee', 'R.Knee', 'L.Ankle', 'R.Ankle'
]

# 骨架连接（用于绘制）
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
    (5, 6),  # 肩膀
    (5, 11), (6, 12),  # 躯干
    (11, 12),
    (5, 7), (7, 9),  # 左臂
    (6, 8), (8, 10),  # 右臂
    (11, 13), (13, 15),  # 左腿
    (12, 14), (14, 16),  # 右腿
]

# 关键点颜色（与COCO一致）
KEYPOINT_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),  # 头部
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170),  # 手臂
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),  # 身体
    (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)  # 腿部
]


# ============ 特征提取 ============
class FeatureExtractor:
    """从YOLOv8 Pose关键点提取10维特征"""
    
    def __init__(self):
        self.prev_features = None
        
    def extract(self, keypoints, frame_shape):
        """
        提取10维特征
        keypoints: (17, 3) - [x, y, confidence]
        frame_shape: (h, w)
        """
        h, w = frame_shape[:2]
        
        # 提取关键点坐标
        nose = keypoints[0]
        l_shoulder = keypoints[5]
        r_shoulder = keypoints[6]
        l_hip = keypoints[11]
        r_hip = keypoints[12]
        l_knee = keypoints[13]
        r_knee = keypoints[14]
        l_ankle = keypoints[15]
        r_ankle = keypoints[16]
        
        # 计算BBox
        valid_points = keypoints[keypoints[:, 2] > 0.3]
        if len(valid_points) < 5:
            return None
        
        x_min, y_min = valid_points[:, :2].min(axis=0)
        x_max, y_max = valid_points[:, :2].max(axis=0)
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        
        # F1: 宽高比
        R = bbox_w / (bbox_h + 1e-6)
        
        # F2: 宽高比变化率
        if self.prev_features is not None:
            delta_R = R - self.prev_features[0]
        else:
            delta_R = 0
        
        # 脊柱中心
        spine_x = (l_shoulder[0] + r_shoulder[0] + l_hip[0] + r_hip[0]) / 4
        spine_y = (l_shoulder[1] + r_shoulder[1] + l_hip[1] + r_hip[1]) / 4
        
        # F3: 脊柱倾角（弧度）
        dx = (l_shoulder[0] + r_shoulder[0]) / 2 - (l_hip[0] + r_hip[0]) / 2
        dy = (l_shoulder[1] + r_shoulder[1]) / 2 - (l_hip[1] + r_hip[1]) / 2
        theta = np.arctan2(dx, -dy) * 180 / np.pi  # 转为度
        
        # F4: 脊柱倾角突变量
        if self.prev_features is not None:
            delta_theta = theta - self.prev_features[2]
        else:
            delta_theta = 0
        
        # 质心
        centroid_x = np.mean(valid_points[:, 0])
        centroid_y = np.mean(valid_points[:, 1])
        yc = centroid_y / h
        
        # F5: 归一化质心y坐标
        F5 = yc
        
        # F6: 质心垂直速度
        if self.prev_features is not None:
            Vy = yc - self.prev_features[4]
        else:
            Vy = 0
        
        # F7: 速度比
        centroid_x_diff = centroid_x / w if self.prev_features is None else (centroid_x - self.prev_features[6]) / w
        Rv = abs(Vy) / (abs(centroid_x_diff) + 1e-6)
        
        # F8: 头脚矢量长度
        head_ankle_dist = np.sqrt((nose[0] - l_ankle[0])**2 + (nose[1] - l_ankle[1])**2)
        Lha = head_ankle_dist / h
        
        # F9: 归一化头部高度
        yh = nose[1] / h
        
        # F10: 脊柱矢量y分量
        spine_vec = np.array([dx, dy])
        spine_len = np.linalg.norm(spine_vec) + 1e-6
        spine_y_norm = abs(dy) / spine_len
        
        features = np.array([R, delta_R, theta, delta_theta, F5, Vy, Rv, Lha, yh, spine_y_norm])
        
        # 归一化（简单的clamp）
        features = np.clip(features, -3, 3)
        
        self.prev_features = [R, delta_R, theta, delta_theta, yc, Vy, centroid_x/w, Rv, yh, spine_y_norm]
        
        return features


class FeatureScaler:
    """特征标准化器，与训练时保持一致"""
    def __init__(self, features_dir="features"):
        self.features_dir = Path(features_dir)
        self.mean_ = None
        self.scale_ = None
    
    def fit_from_train(self):
        """从训练集特征计算mean和std"""
        train_X = np.load(self.features_dir / "train" / "X.npy")
        # 展平为 (N*20, 10) 再计算统计量
        train_2d = train_X.reshape(-1, train_X.shape[-1])
        self.mean_ = train_2d.mean(axis=0)
        self.scale_ = train_2d.std(axis=0)
        print(f"  ✓ StandardScaler 参数已计算 (from {train_X.shape})")
        print(f"    mean范围: [{self.mean_.min():.4f}, {self.mean_.max():.4f}]")
        print(f"    std范围: [{self.scale_.min():.4f}, {self.scale_.max():.4f}]")
    
    def transform(self, X):
        """标准化特征"""
        if self.mean_ is None:
            self.fit_from_train()
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = (X_flat - self.mean_) / (self.scale_ + 1e-8)
        return X_scaled.reshape(X.shape)


# ============ 模型加载 ============
def load_model(checkpoint_path):
    """加载模型，返回模型和config（scaler参数不再从这里取）"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    
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

    # 不再依赖checkpoint里的scaler参数，改由FeatureScaler从训练集计算
    return model, config


def get_fall_probability(model, X_scaled):
    """获取Fall概率"""
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)[0, 1].numpy()
    return probs


# ============ 可视化 ============
class Visualizer:
    """可视化模块"""
    
    def __init__(self, frame_shape):
        h, w = frame_shape[:2]
        self.h, self.w = h, w
        
        # 概率历史（用于绘制曲线）
        self.prob_history = deque(maxlen=60)  # 保留60帧历史
        
        # 信息面板
        self.info_font = cv2.FONT_HERSHEY_SIMPLEX
        self.info_size = 0.5
        self.info_thickness = 1
        
    def draw_skeleton(self, frame, keypoints):
        """绘制骨架图"""
        # 绘制连接线
        for idx1, idx2 in SKELETON_CONNECTIONS:
            pt1 = keypoints[idx1]
            pt2 = keypoints[idx2]
            if pt1[2] > 0.3 and pt2[2] > 0.3:  # 置信度阈值
                cv2.line(frame, 
                         (int(pt1[0]), int(pt1[1])), 
                         (int(pt2[0]), int(pt2[1])), 
                         (0, 255, 0), 2)
        
        # 绘制关键点
        for i, kp in enumerate(keypoints):
            if kp[2] > 0.3:
                color = KEYPOINT_COLORS[i]
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, color, -1)
                # 标注编号
                cv2.putText(frame, str(i), (int(kp[0])+5, int(kp[1])-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def draw_bbox(self, frame, keypoints, aspect_ratio):
        """绘制BBox和宽高比"""
        valid_points = keypoints[keypoints[:, 2] > 0.3]
        if len(valid_points) < 5:
            return
        
        x_min, y_min = valid_points[:, :2].min(axis=0).astype(int)
        x_max, y_max = valid_points[:, :2].max(axis=0).astype(int)
        
        # 绘制矩形框
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # 显示宽高比
        label = f"R={aspect_ratio:.2f}"
        cv2.putText(frame, label, (x_min, y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def draw_angle(self, frame, keypoints):
        """绘制脊柱倾角"""
        l_shoulder = keypoints[5]
        r_shoulder = keypoints[6]
        l_hip = keypoints[11]
        r_hip = keypoints[12]
        
        if all(kp[2] > 0.3 for kp in [l_shoulder, r_shoulder, l_hip, r_hip]):
            # 计算角度
            shoulder_mid_x = (l_shoulder[0] + r_shoulder[0]) / 2
            shoulder_mid_y = (l_shoulder[1] + r_shoulder[1]) / 2
            hip_mid_x = (l_hip[0] + r_hip[0]) / 2
            hip_mid_y = (l_hip[1] + r_hip[1]) / 2
            
            # 绘制脊柱线
            cv2.line(frame, 
                    (int(shoulder_mid_x), int(shoulder_mid_y)),
                    (int(hip_mid_x), int(hip_mid_y)),
                    (255, 255, 0), 2)
            
            # 计算并显示角度
            dx = shoulder_mid_x - hip_mid_x
            dy = shoulder_mid_y - hip_mid_y
            angle = np.arctan2(dx, -dy) * 180 / np.pi
            
            # 绘制角度弧
            center = (int(hip_mid_x), int(hip_mid_y))
            cv2.putText(frame, f"θ={angle:.1f}°", (center[0]+10, center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def draw_prob_curve(self, frame, prob):
        """绘制概率变化曲线"""
        self.prob_history.append(prob)
        
        h, w = self.h, self.w
        curve_height = 80
        curve_width = 200
        margin_x = 10
        margin_y = h - curve_height - 10
        
        # 背景面板
        panel = np.zeros((curve_height, curve_width, 3), dtype=np.uint8)
        cv2.rectangle(panel, (0, 0), (curve_width, curve_height), (40, 40, 40), -1)
        cv2.rectangle(panel, (0, 0), (curve_width, curve_height), (100, 100, 100), 1)
        
        # 绘制曲线
        if len(self.prob_history) > 1:
            points = []
            for i, p in enumerate(self.prob_history):
                x = int(i * curve_width / 60)
                y = curve_height - int(p * (curve_height - 10))
                points.append((x, y))
            
            for i in range(len(points) - 1):
                color = (0, 0, 255) if self.prob_history[-1] > 0.5 else (0, 255, 0)
                cv2.line(panel, points[i], points[i+1], color, 2)
        
        # 阈值线
        threshold_y = curve_height - int(DEFAULT_THRESHOLD * (curve_height - 10))
        cv2.line(panel, (0, threshold_y), (curve_width, threshold_y), (128, 128, 128), 1)
        cv2.putText(panel, f"thr={DEFAULT_THRESHOLD}", (5, threshold_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
        
        # 标签
        cv2.putText(panel, "Fall Prob", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # 放到frame上
        frame[margin_y:margin_y+curve_height, margin_x:margin_x+curve_width] = panel
        
        # 当前概率值
        cv2.putText(frame, f"P={prob:.2f}", (margin_x + 5, margin_y + curve_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if prob < 0.5 else (0, 0, 255), 2)
    
    def draw_fall_status(self, frame, is_fall, prob):
        """右上角显示摔倒状态"""
        if is_fall:
            text = "FALL DETECTED!"
            color = (0, 0, 255)  # 红色
            # 绘制背景
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            cv2.rectangle(frame, 
                         (self.w - text_size[0] - 20, 10), 
                         (self.w - 10, 10 + text_size[1] + 20),
                         (0, 0, 200), -1)
            cv2.putText(frame, text, (self.w - text_size[0] - 10, 10 + text_size[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        else:
            text = "Normal"
            color = (0, 200, 0)  # 绿色
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame,
                         (self.w - text_size[0] - 20, 10),
                         (self.w - 10, 10 + text_size[1] + 15),
                         (0, 150, 0), -1)
            cv2.putText(frame, text, (self.w - text_size[0] - 10, 10 + text_size[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def draw_info(self, frame, frame_idx, total_frames, fps):
        """左下角显示信息"""
        info_text = f"Frame: {frame_idx}/{total_frames} | FPS: {fps:.1f}"
        cv2.putText(frame, info_text, (10, self.h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ============ 主函数 ============
def play_video(cap, video_path, model_a, model_b, pose_model,
               feature_scaler,
               window_name, feature_extractor, visualizer,
               stride, threshold, speed_factor):
    """播放单个视频"""
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到开头
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    video_name = Path(video_path).name
    
    # 重置组件状态
    feature_extractor.prev_features = None
    visualizer.prob_history.clear()
    
    # 特征窗口缓冲区
    feature_buffer = deque(maxlen=WINDOW_SIZE)
    
    
    # ============ 预加载阶段：仅预热模型，不填充特征缓冲区 ============
    # 读取一帧并运行 YOLO 来预热 CUDA / 模型
    ret, warmup_frame = cap.read()
    if ret and pose_model is not None:
        _ = pose_model(warmup_frame, verbose=False)

    # 重置到开头（正式播放）
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    feature_extractor.prev_features = None  # 重置特征追踪状态
    feature_buffer.clear()                   # 确保缓冲区为空，在播放时按正确 stride 填充
    
    # 跳帧计数器
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                return 'end'  # 视频播放完毕
            
            frame_count += 1
            
            # 更新窗口标题
            cv2.setWindowTitle(window_name, f"Fall Detection - {video_name} ({frame_count}/{total_frames})")
            
            # 跳帧处理
            if frame_count % stride != 0:
                cv2.imshow(window_name, frame)
                wait_time = int(1000 / fps / speed_factor)
                key = cv2.waitKey(wait_time) & 0xFF
                if key == ord('q'):
                    return 'quit'
                elif key == ord('n'):
                    return 'next'  # 切换下一个
                elif key == ord('p'):
                    paused = not paused
                continue
            
            # ============ 姿态估计 ============
            kpts = None
            if pose_model is not None:
                results = pose_model(frame, verbose=False)
                result = results[0]
                
                if result.keypoints is not None and len(result.keypoints) > 0:
                    kpts = result.keypoints[0].data.cpu().numpy()[0]
            
            # ============ 特征提取 ============
            current_feature = None
            if kpts is not None:
                current_feature = feature_extractor.extract(kpts, frame.shape)
            
            if current_feature is not None:
                feature_buffer.append(current_feature)
                
                if len(feature_buffer) == WINDOW_SIZE:
                    X = np.array(feature_buffer)
                    
                    # === 特征标准化（与训练时一致，使用训练集统计量）===
                    X = feature_scaler.transform(X)
                    X = np.clip(X, -3, 3)
                    
                    prob_a = get_fall_probability(model_a, X)
                    
                    # 模型B使用同一套标准化参数（同一训练集）
                    X_b = np.array(feature_buffer)
                    X_b = feature_scaler.transform(X_b)
                    X_b = np.clip(X_b, -3, 3)
                    prob_b = get_fall_probability(model_b, X_b)
                    
                    ensemble_prob = ENSEMBLE_WEIGHT_A * prob_a + ENSEMBLE_WEIGHT_B * prob_b
                    is_fall = ensemble_prob >= threshold
                    
                    # 可视化
                    visualizer.draw_skeleton(frame, kpts)
                    visualizer.draw_bbox(frame, kpts, current_feature[0])
                    visualizer.draw_angle(frame, kpts)
                    visualizer.draw_prob_curve(frame, ensemble_prob)
                    visualizer.draw_fall_status(frame, is_fall, ensemble_prob)
                    status = "FALL!" if is_fall else "Normal"
                else:
                    status = "Collecting..."
            else:
                status = "No Person"
            
            visualizer.draw_info(frame, frame_count, total_frames, fps)
            cv2.imshow(window_name, frame)
        
        wait_time = int(1000 / fps / speed_factor)
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord('q'):
            return 'quit'
        elif key == ord('n'):
            return 'next'
        elif key == ord('p'):
            paused = not paused


def main():
    import argparse
    parser = argparse.ArgumentParser(description="摔倒检测演示")
    parser.add_argument('--video', type=str, 
                       default=str(BASE_DIR / "dataset" / "test" / "fall" / "fall-07-cam0.mp4"),
                       help='视频文件路径')
    parser.add_argument('--dir', type=str, default=None,
                       help='视频目录路径（播放目录下所有视频）')
    parser.add_argument('--playlist', type=str, default=None,
                       help='自定义播放列表，格式: fall-07,adl-07,...（使用dataset/test下的相对路径）')
    parser.add_argument('--demo', action='store_true', default=True,
                       help='使用默认8视频交叉播放列表（4个fall + 4个adl，默认开启）')
    parser.add_argument('--no-demo', action='store_true',
                       help='禁用演示模式，使用单视频或目录模式')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                       help=f'决策阈值 (默认={DEFAULT_THRESHOLD})')
    parser.add_argument('--stride', type=int, default=WINDOW_STRIDE,
                       help=f'跳帧间隔 (默认={WINDOW_STRIDE})')
    parser.add_argument('--speed', type=float, default=0.5,
                       help='播放速度 (0.5=0.5倍速, 默认=0.5)')
    args = parser.parse_args()
    
    print("="*60)
    print("摔倒检测系统 - 演示模式")
    print("="*60)
    
    # 获取视频列表
    video_dir = None
    video_files = []
    
    # 默认使用演示模式（除非指定 --no-demo 或 --dir 或 --video）
    use_demo = args.demo and not args.no_demo and not args.dir and args.playlist is None
    
    if use_demo:
        video_files = [f for f in DEMO_PLAYLIST if f.exists()]
        if video_files:
            print(f"\n演示模式: 共 {len(video_files)} 个视频（前2 + 后2）")
            for i, f in enumerate(video_files):
                label = "FALL" if "fall" in f.name.lower() else "ADL"
                print(f"  [{i+1}] [{label}] {f.name}")
        else:
            print("错误: 演示播放列表为空，请检查视频文件是否存在")
            return
    elif args.playlist is not None:
        # 播放列表模式：交叉排列 4个fall + 4个adl
        if args.playlist == "":
            # 空参数，使用默认播放列表
            video_files = [f for f in DEMO_PLAYLIST if f.exists()]
        else:
            # 格式: fall-07,adl-07,fall-10,adl-10,...（自动添加-cam0后缀）
            names = [n.strip() for n in args.playlist.split(',')]
            for name in names:
                # 自动判断是fall还是adl，并添加-cam0后缀
                if name.startswith('fall-'):
                    video_path = BASE_DIR / "dataset" / "test" / "fall" / f"{name}-cam0.mp4"
                elif name.startswith('adl-'):
                    video_path = BASE_DIR / "dataset" / "test" / "adl" / f"{name}-cam0.mp4"
                else:
                    video_path = BASE_DIR / "dataset" / "test" / "fall" / f"{name}-cam0.mp4"
                    if not video_path.exists():
                        video_path = BASE_DIR / "dataset" / "test" / "adl" / f"{name}-cam0.mp4"
                
                if video_path.exists():
                    video_files.append(video_path)
                else:
                    print(f"警告: 视频文件不存在，跳过: {name}")
        
        if video_files:
            print(f"\n播放列表模式: 共 {len(video_files)} 个视频（交叉排列）")
            for i, f in enumerate(video_files):
                label = "FALL" if "fall" in f.name.lower() else "ADL"
                print(f"  [{i+1}] [{label}] {f.name}")
        else:
            print("错误: 播放列表为空")
            return
    else:
        video_dir = Path(args.dir) if args.dir else None
        
        if args.dir and video_dir.exists():
            video_files = sorted([f for f in video_dir.glob("*.mp4")])
            if video_files:
                print(f"\n目录: {video_dir}")
                print(f"找到 {len(video_files)} 个视频文件:")
                for i, f in enumerate(video_files):
                    print(f"  [{i+1}] {f.name}")
            else:
                print(f"错误: 目录中没有找到MP4文件: {video_dir}")
                return
        elif args.dir:
            print(f"错误: 目录不存在: {video_dir}")
            return
        else:
            # 单视频模式
            video_path = Path(args.video)
            if not video_path.exists():
                print(f"错误: 视频文件不存在: {video_path}")
                return
            video_files = [video_path]
    
    # 加载模型
    print("\n加载模型...")
    print(f"  模型A (高精确率): {CKPT_MODEL_A.name}")
    model_a, config_a = load_model(CKPT_MODEL_A)
    print(f"  模型B (高召回率): {CKPT_MODEL_B.name}")
    model_b, config_b = load_model(CKPT_MODEL_B)
    print(f"  集成决策阈值: {args.threshold}")
    
    # 初始化特征标准化器（从训练集计算统计量）
    print("\n初始化特征标准化器...")
    feature_scaler = FeatureScaler(BASE_DIR / "features")
    feature_scaler.fit_from_train()
    
    # 初始化YOLOv8 Pose
    pose_model = None
    if YOLO_AVAILABLE:
        print("\n加载YOLOv8 Pose模型...")
        pose_model = YOLO('yolov8n-pose.pt')
    
    # 初始化窗口
    window_name = 'Fall Detection Demo'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 初始化组件
    feature_extractor = FeatureExtractor()
    
    # 播放视频列表
    current_idx = 0
    
    print("\n" + "="*60)
    print("控制说明:")
    print("  q - 退出")
    print("  n - 切换下一个视频")
    print("  p - 暂停/继续")
    print("="*60)
    
    while current_idx < len(video_files):
        video_path = video_files[current_idx]
        
        # 打开视频获取尺寸
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"错误: 无法打开视频: {video_path}")
            current_idx += 1
            continue
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 调整窗口大小
        cv2.resizeWindow(window_name, min(width, 1280), min(height, 720))
        
        # 创建新的visualizer
        visualizer = Visualizer((height, width))
        
        print(f"\n▶ 播放 [{current_idx+1}/{len(video_files)}]: {video_path.name}")
        
        # 播放视频
        result = play_video(
            cap, video_path, model_a, model_b, pose_model,
            feature_scaler,
            window_name, feature_extractor, visualizer,
            args.stride, args.threshold, args.speed
        )
        
        cap.release()
        
        if result == 'quit':
            print("\n退出演示")
            break
        elif result == 'next':
            current_idx += 1
            if current_idx >= len(video_files):
                print("\n已播放完所有视频")
                break
    
    cv2.destroyAllWindows()
    print("\n演示结束")


if __name__ == "__main__":
    main()
