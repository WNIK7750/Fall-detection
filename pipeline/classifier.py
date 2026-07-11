"""
3-Class Fall Classifier — wraps EnhancedFallDetector with dual-threshold logic.

Model: BiLSTM + BiGRU + Bidirectional Multi-Head Self-Attention (~4.3M params)
Input: (batch, 20, 10) — 10-D features over 20-frame sliding window
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

        # Build model
        self.model = build_enhanced_model(
            input_size=input_size,
            hidden_size=128,
            variant="full",
        )
        self._load_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoint(self, path: str):
        """Load checkpoint, handling various key formats."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        # Handle module. prefix from DataParallel
        if any(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        self.model.load_state_dict(state, strict=False)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict fall probability.

        Args:
            features: (N, 20, 10) or (20, 10) — 10-D features over 20 timesteps.
        Returns:
            (N, 2) or (2,) probabilities [p_normal, p_fall].
        """
        single = (features.ndim == 2)
        x = features[np.newaxis, ...] if single else features
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


# --- Quick test with dummy data ---
if __name__ == "__main__":
    print("FallClassifier module ready.")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  Model: BiLSTM+BiGRU+Attention (~4.3M params)")

    # Generate dummy features
    normal_features = np.random.randn(1, 20, 10).astype(np.float32)
    normal_features[:, :, 2] *= 0.3  # small torso angle
    normal_features[:, :, 4] += 1.0  # high COM

    print("  Note: Module loaded. Requires trained checkpoint for inference.")
    print("  Run extract_dataset_3d.py + train_3d.py first.")
