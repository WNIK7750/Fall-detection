"""
=============================================================================
摔倒检测系统 - 特征提取模块
适配 MediaPipe 0.10.35+ 新版 Tasks API
=============================================================================
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

# ============ 尝试导入 MediaPipe ============
mp_available = False

try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    mp_available = True
    print(f"MediaPipe Tasks API 可用")
except ImportError:
    print("警告: MediaPipe Tasks API 不可用")

# ============ 配置 ============
class Config:
    # MediaPipe Pose 关键点索引
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    # 用于特征提取的关键点索引
    KEYPOINT_IDX = [
        NOSE, LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
        LEFT_ANKLE, RIGHT_ANKLE
    ]

    # 跳帧间隔
    SKIP_FRAMES = 3
    
    # 滑动窗口参数
    WINDOW_SIZE = 20
    WINDOW_STRIDE = 5
    
    # 特征归一化参数
    FEATURE_MIN = np.array([0.0, -0.5, 0.0, -30, 0.2, -0.1, 0.0, 0.1, 0.1, -1.0])
    FEATURE_MAX = np.array([2.0, 0.5, 90, 30, 0.8, 0.1, 20, 1.0, 0.5, 1.0])


class MediaPipePoseExtractor:
    """
    基于 MediaPipe Tasks API 的姿态提取器
    兼容 MediaPipe 0.10.35+
    """
    
    def __init__(self):
        """初始化 MediaPipe Pose Landmarker"""
        self.frame_idx = 0
        
        if not mp_available:
            raise RuntimeError("MediaPipe Tasks API 不可用")
        
        # 模型文件路径
        model_path = Path(r"D:\识别人类摔倒的智能系统\models\pose_landmarker_lite.task")
        
        # 如果模型不存在，尝试使用另一种初始化方式
        if not model_path.exists():
            print(f"模型文件不存在: {model_path}")
            print("尝试使用其他初始化方式...")
            
            # 尝试不指定模型（使用打包的默认模型）
            try:
                options = vision.PoseLandmarkerOptions(
                    base_options=None,  # 不指定模型，使用内置
                    running_mode=vision.RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.detector = vision.PoseLandmarker.create_from_options(options)
                print("使用默认模型初始化成功")
            except Exception as e:
                print(f"初始化失败: {e}")
                print()
                print("请先运行以下命令下载模型:")
                print("  python download_mediapipe_model.py")
                raise RuntimeError("模型文件缺失")
        else:
            # 使用指定模型文件
            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)
            print(f"使用模型文件: {model_path}")
        
        print("MediaPipe Pose Landmarker 初始化完成")
    
    def detect(self, rgb_frame, timestamp_ms):
        """
        检测姿态
        
        Args:
            rgb_frame: RGB格式图像
            timestamp_ms: 时间戳（毫秒）
            
        Returns:
            关键点列表 or None
        """
        # 转换为 MediaPipe Image
        mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=rgb_frame)
        
        # 检测
        result = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        if not result.pose_landmarks:
            return None
        
        return result.pose_landmarks[0]
    
    def extract_keypoints(self, frame, timestamp_ms=0):
        """
        从帧中提取关键点
        
        Returns:
            dict: {'keypoints': np.array, 'confidences': np.array} or None
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = self.detect(rgb_frame, timestamp_ms)
        
        if landmarks is None:
            return None
        
        keypoints = []
        confidences = []
        
        for idx in Config.KEYPOINT_IDX:
            kp = landmarks[idx]
            keypoints.append([kp.x, kp.y, kp.z])
            confidences.append(kp.visibility)
        
        return {
            'keypoints': np.array(keypoints),
            'confidences': np.array(confidences)
        }
    
    def close(self):
        """关闭检测器"""
        if hasattr(self, 'detector'):
            self.detector.close()


class FeatureExtractor:
    """10维特征提取器"""
    
    def __init__(self):
        if mp_available:
            self.pose_extractor = MediaPipePoseExtractor()
        else:
            self.pose_extractor = None
        
        self.prev_features = None
    
    def compute_bbox(self, keypoints, confidences):
        """计算人体外接矩形框"""
        valid_mask = confidences > 0.5
        if not np.any(valid_mask):
            return None, None
        
        valid_points = keypoints[valid_mask]
        x_min, y_min = valid_points[:, :2].min(axis=0)
        x_max, y_max = valid_points[:, :2].max(axis=0)
        
        width = x_max - x_min
        height = y_max - y_min
        
        return np.array([x_min, y_min, x_max, y_max]), width / (height + 1e-6)
    
    def compute_spine_angle(self, keypoints):
        """计算脊柱倾角"""
        left_shoulder = keypoints[1][:2]
        right_shoulder = keypoints[2][:2]
        left_hip = keypoints[3][:2]
        right_hip = keypoints[4][:2]
        
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        hip_mid = (left_hip + right_hip) / 2
        
        spine_vec = shoulder_mid - hip_mid
        angle = np.abs(np.degrees(np.arctan2(spine_vec[0], -spine_vec[1])))
        
        return angle, spine_vec
    
    def compute_centroid(self, keypoints, confidences):
        """计算人体质心"""
        valid_mask = confidences > 0.5
        if not np.any(valid_mask):
            return None, None
        
        valid_points = keypoints[valid_mask, :2]
        valid_confs = confidences[valid_mask]
        
        weights = valid_confs / (valid_confs.sum() + 1e-6)
        centroid = np.average(valid_points, axis=0, weights=weights)
        
        return centroid, valid_mask
    
    def extract_features(self, frame, timestamp_ms=0):
        """从单帧提取10维特征"""
        if self.pose_extractor is None:
            return np.zeros(10) if self.prev_features is None else self.prev_features * 0.8
        
        kp_data = self.pose_extractor.extract_keypoints(frame, timestamp_ms)
        
        if kp_data is None:
            return np.zeros(10) if self.prev_features is None else self.prev_features * 0.8
        
        keypoints = kp_data['keypoints']
        confidences = kp_data['confidences']
        
        features = np.zeros(10)
        
        # F1: 宽高比 R
        bbox, R = self.compute_bbox(keypoints, confidences)
        if R is not None:
            features[0] = np.clip(R, 0.1, 3.0)
        
        # F2: 宽高比变化率 ΔR
        if self.prev_features is not None and self.prev_features[0] > 0:
            features[1] = features[0] - self.prev_features[0]
        
        # F3, F4: 脊柱倾角 θ 和突变量 Δθ
        theta, spine_vec = self.compute_spine_angle(keypoints)
        features[2] = theta
        if self.prev_features is not None and self.prev_features[2] > 0:
            features[3] = theta - self.prev_features[2]
        
        # F5: 归一化质心y坐标
        centroid, valid_mask = self.compute_centroid(keypoints, confidences)
        if centroid is not None:
            features[4] = centroid[1]
        
        # F6: 质心垂直速度
        if self.prev_features is not None and self.prev_features[4] > 0:
            features[5] = features[4] - self.prev_features[4]
        
        # F7: 速度比 R_v
        if len(keypoints) > 3 and self.prev_features is not None:
            vy = features[5]
            vx = features[1] * 10
            features[6] = np.abs(vy) / (np.abs(vx) + 0.01)
        
        # F8: 头脚矢量归一化长度
        nose = keypoints[0][:2]
        left_ankle = keypoints[7][:2]
        right_ankle = keypoints[8][:2]
        mid_ankle = (left_ankle + right_ankle) / 2
        head_ankle_dist = np.linalg.norm(nose - mid_ankle)
        features[7] = head_ankle_dist
        
        # F9: 归一化头部高度
        features[8] = nose[1]
        
        # F10: 脊柱矢量y分量
        if np.linalg.norm(spine_vec) > 1e-6:
            features[9] = -spine_vec[1] / (np.linalg.norm(spine_vec) + 1e-6)
        
        self.prev_features = features
        return features
    
    def normalize_features(self, features):
        """Min-Max归一化"""
        normalized = (features - Config.FEATURE_MIN) / (Config.FEATURE_MAX - Config.FEATURE_MIN + 1e-6)
        return np.clip(normalized, 0, 1)
    
    def close(self):
        """关闭资源"""
        if self.pose_extractor:
            self.pose_extractor.close()


def process_video(video_path, extractor):
    """处理单个视频"""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0
    
    features_list = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 跳帧处理
        if frame_count % Config.SKIP_FRAMES != 0:
            continue
        
        # 计算时间戳
        timestamp_ms = int((frame_count / fps) * 1000)
        
        # 提取特征
        features = extractor.extract_features(frame, timestamp_ms)
        normalized_features = extractor.normalize_features(features)
        
        features_list.append(normalized_features)
    
    cap.release()
    
    if len(features_list) == 0:
        return None
    
    return np.array(features_list)


def create_windows(features, window_size=20, stride=5, label=None):
    """创建滑动窗口样本"""
    windows = []
    labels = []
    
    for i in range(0, len(features) - window_size + 1, stride):
        window = features[i:i + window_size]
        windows.append(window)
        if label is not None:
            labels.append(label)
    
    windows = np.array(windows)
    
    if label is not None:
        labels = np.array(labels)
        return windows, labels
    
    return windows


def extract_dataset(input_dir, output_dir, extractor):
    """批量处理数据集"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_features = []
    all_labels = []
    
    # 处理摔倒视频
    fall_dir = input_path / "fall"
    if fall_dir.exists():
        fall_videos = sorted(fall_dir.glob("*.mp4"))
        print(f"\n处理摔倒视频: {len(fall_videos)} 个")
        
        for video_path in tqdm(fall_videos, desc="摔倒视频"):
            features = process_video(video_path, extractor)
            if features is not None:
                windows, labels = create_windows(
                    features, Config.WINDOW_SIZE, Config.WINDOW_STRIDE, label=1
                )
                all_features.append(windows)
                all_labels.append(labels)
    
    # 处理正常视频
    adl_dir = input_path / "adl"
    if adl_dir.exists():
        adl_videos = sorted(adl_dir.glob("*.mp4"))
        print(f"\n处理正常视频: {len(adl_videos)} 个")
        
        for video_path in tqdm(adl_videos, desc="正常视频"):
            features = process_video(video_path, extractor)
            if features is not None:
                windows, labels = create_windows(
                    features, Config.WINDOW_SIZE, Config.WINDOW_STRIDE, label=0
                )
                all_features.append(windows)
                all_labels.append(labels)
    
    if all_features:
        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        
        np.save(output_path / "X.npy", X)
        np.save(output_path / "y.npy", y)
        
        print(f"\n特征提取完成!")
        print(f"  样本数: {len(X)}")
        print(f"  特征维度: {X.shape[1]}帧 × {X.shape[2]}维")
        print(f"  摔倒样本: {np.sum(y == 1)}")
        print(f"  正常样本: {np.sum(y == 0)}")
        print(f"  保存到: {output_path}")
        
        return X, y
    
    return None, None


def main():
    print("=" * 60)
    print("摔倒检测系统 - 特征提取模块")
    print("MediaPipe 0.10.35+ Tasks API 兼容版")
    print("=" * 60)
    
    if not mp_available:
        print("错误: MediaPipe Tasks API 不可用")
        print("请运行: pip install mediapipe>=0.10.30")
        return
    
    base_dir = Path(r"D:\识别人类摔倒的智能系统")
    dataset_dir = base_dir / "dataset"
    features_dir = base_dir / "features"
    
    extractor = FeatureExtractor()
    
    try:
        print("\n>>> 处理训练集...")
        X_train, y_train = extract_dataset(
            dataset_dir / "train",
            features_dir / "train",
            extractor
        )
        
        print("\n>>> 处理验证集...")
        X_val, y_val = extract_dataset(
            dataset_dir / "val",
            features_dir / "val",
            extractor
        )
        
        print("\n>>> 处理测试集...")
        X_test, y_test = extract_dataset(
            dataset_dir / "test",
            features_dir / "test",
            extractor
        )
        
        stats = {
            "train": {"samples": int(len(X_train)) if X_train is not None else 0},
            "val": {"samples": int(len(X_val)) if X_val is not None else 0},
            "test": {"samples": int(len(X_test)) if X_test is not None else 0},
            "config": {
                "window_size": Config.WINDOW_SIZE,
                "window_stride": Config.WINDOW_STRIDE,
                "skip_frames": Config.SKIP_FRAMES,
                "feature_dim": 10
            }
        }
        
        with open(features_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print("\n" + "=" * 60)
        print("特征提取全部完成!")
        print("=" * 60)
        
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
