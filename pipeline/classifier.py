"""
3-Class Fall Classifier — wraps EnhancedFallDetector with dual-threshold logic.

Model: BiLSTM + BiGRU + Bidirectional Multi-Head Self-Attention (~4.3M params)
Input: (batch, sequence, 10) — 10-D features over a temporal window
Output: 3 classes via dual thresholds on fall probability.

Decision logic:
    P(fall) > threshold_high    → "fall"     (class 2)
    threshold_low < P ≤ high    → "suspected" (class 1)
    P(fall) ≤ threshold_low     → "normal"    (class 0)

Thresholds are configurable, default 0.4/0.85.
"""

import os
import sys
import numpy as np
import torch

# Import existing model architecture
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.enhanced_lstm import EnhancedFallDetector, build_enhanced_model


class FallClassifier:
    """3-class fall detector with configurable thresholds."""

    CLASS_NAMES = {0: "normal", 1: "suspected", 2: "fall"}

    def __init__(
        self,
        checkpoint_path: str,
        input_size: int = 10,
        threshold_low: float = 0.4,
        threshold_high: float = 0.85,
        device: str | None = None,
    ):
        """
        Args:
            checkpoint_path: Path to trained .pth checkpoint.
            input_size: Feature dimension (10 for our 3D features).
            threshold_low: Below this → "normal".
            threshold_high: Above this → "fall". Between → "suspected".
            device: 'cuda', 'cpu', or None (auto).
        """
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        ckpt = self._load_checkpoint_data(checkpoint_path)
        input_size = int(ckpt.get("input_size", input_size)) if isinstance(ckpt, dict) else input_size
        self.sequence_length = (
            int(ckpt["sequence_length"])
            if isinstance(ckpt, dict) and ckpt.get("sequence_length") is not None
            else None
        )

        # Build model
        self.model = build_enhanced_model(
            input_size=input_size,
            hidden_size=128,
            model_type="full",
        )
        self._load_checkpoint(ckpt)
        self.model.to(self.device)
        self.model.eval()

        self.scaler_mean: np.ndarray | None = None
        self.scaler_scale: np.ndarray | None = None
        if isinstance(ckpt, dict):
            mean = ckpt.get("scaler_mean")
            scale = ckpt.get("scaler_scale", ckpt.get("scaler_std"))
            if mean is not None and scale is not None:
                self.scaler_mean = np.asarray(mean, dtype=np.float32)
                self.scaler_scale = np.asarray(scale, dtype=np.float32)
                self.scaler_scale = np.where(self.scaler_scale == 0, 1.0, self.scaler_scale)

    def _load_checkpoint_data(self, path: str):
        """Load raw checkpoint data."""
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return torch.load(path, map_location=self.device, weights_only=False)

    def _load_checkpoint(self, ckpt):
        """Load checkpoint, handling various key formats."""
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        # Handle module. prefix from DataParallel
        if any(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        """Apply training StandardScaler parameters if present in checkpoint."""
        if self.scaler_mean is None or self.scaler_scale is None:
            return features
        if features.shape[-1] != len(self.scaler_mean):
            raise ValueError(
                f"Feature dimension {features.shape[-1]} does not match scaler "
                f"dimension {len(self.scaler_mean)}"
            )
        return (features - self.scaler_mean) / self.scaler_scale

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict fall probability.

        Args:
            features: (N, T, 10) or (T, 10), currently T=40.
        Returns:
            (N, 2) or (2,) probabilities [p_normal, p_fall].
        """
        single = (features.ndim == 2)
        x = features[np.newaxis, :, :] if single else features
        if self.sequence_length is not None and x.shape[1] != self.sequence_length:
            raise ValueError(
                f"Sequence length {x.shape[1]} does not match checkpoint "
                f"sequence length {self.sequence_length}"
            )
        x = self._standardize(x.astype(np.float32, copy=False))
        x = torch.tensor(x, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            result = probs.cpu().numpy()

        return result[0] if single else result

    def predict(self, features: np.ndarray) -> dict:
        """Predict class label with probability.

        Returns:
            dict with keys: class_id, class_name, fall_prob, thresholds.
        """
        probs = self.predict_proba(features)
        p_fall = float(probs[1])

        if p_fall > self.threshold_high:
            class_id, name = 2, "fall"
        elif p_fall > self.threshold_low:
            class_id, name = 1, "suspected"
        else:
            class_id, name = 0, "normal"

        return {
            "class_id": class_id,
            "class_name": name,
            "fall_prob": round(p_fall, 4),
            "normal_prob": round(float(probs[0]), 4),
            "threshold_low": self.threshold_low,
            "threshold_high": self.threshold_high,
        }

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


class HeuristicClassifier:
    """Rule-based fall detector using 3D biomechanical features.

    Uses simple thresholds on COM height and spine angle as a fallback
    when a trained ML classifier is not available. Honest and transparent.
    """

    CLASS_NAMES = {0: "normal", 1: "suspected", 2: "fall"}

    def __init__(
        self,
        com_ratio_threshold: float = 0.55,
        spine_z_threshold: float = 0.6,
        velocity_threshold: float = 0.2,
    ):
        """
        Args:
            com_ratio_threshold: COM height / standing_height below this → potential fall.
            spine_z_threshold: Spine Z component below this → torso is near horizontal.
            velocity_threshold: Vertical velocity magnitude above this → rapid movement.
        """
        self.com_th = com_ratio_threshold
        self.spine_th = spine_z_threshold
        self.vel_th = velocity_threshold

    def predict(self, features: np.ndarray) -> dict:
        """Heuristic fall detection from 10-D features.

        Features (indices):
            0: AR, 1: dAR, 2: torso_angle(°), 3: d_torso,
            4: COM_height_ratio, 5: COM_vel_Z, 6: vel_ratio,
            7: head_foot_dist, 8: head_height_ratio, 9: spine_Z

        Rules:
            COM near floor + spine horizontal → fall
            COM dropping fast or low → suspected
            Otherwise → normal
        """
        if features.ndim == 2:
            # (T, 10) -> take mean over window
            f = features.mean(axis=0)
        else:
            f = features

        com_h = float(f[4])       # COM height / standing height
        spine_z = float(f[9])     # spine direction Z (1=upright)
        com_vel = float(f[5])     # vertical velocity
        torso_angle = float(f[2]) # degrees from upright

        # Fall: low COM + near-horizontal spine
        is_fall = (com_h < self.com_th and abs(spine_z) < self.spine_th)
        # Suspected: dropping fast or low COM
        is_suspected = abs(com_vel) > self.vel_th or com_h < self.com_th * 1.5

        if is_fall:
            class_id, name = 2, "fall"
        elif is_suspected:
            class_id, name = 1, "suspected"
        else:
            class_id, name = 0, "normal"

        return {
            "class_id": class_id,
            "class_name": name,
            "fall_prob": round(0.95 if is_fall else (0.6 if is_suspected else 0.1), 2),
            "normal_prob": round(0.05 if is_fall else (0.4 if is_suspected else 0.9), 2),
            "com_height": round(com_h, 3),
            "spine_z": round(spine_z, 3),
            "torso_angle": round(torso_angle, 1),
        }


# --- Quick test with dummy data ---
if __name__ == "__main__":
    print("FallClassifier module ready.")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  Model: BiLSTM+BiGRU+Attention (~4.3M params)")

    # Generate dummy features
    normal_features = np.random.randn(1, 40, 10).astype(np.float32)
    normal_features[:, :, 2] *= 0.3  # small torso angle
    normal_features[:, :, 4] += 1.0  # high COM

    print("  Note: Module loaded. Requires trained checkpoint for inference.")
    print("  Run extract_dataset_3d.py + train_3d.py first.")
