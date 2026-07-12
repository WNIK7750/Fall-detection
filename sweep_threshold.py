"""Threshold sweep for the 3D fall classifier."""

import argparse
import os

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from pipeline.classifier import FallClassifier


DEFAULT_CHECKPOINT = os.path.join(
    "checkpoints",
    "enhanced_3d_20260712_185406",
    "best_model.pth",
)


def parse_thresholds(raw: str | None, start: float, stop: float, step: float) -> list[float]:
    if raw:
        return [float(x.strip()) for x in raw.split(",") if x.strip()]
    values = []
    t = start
    while t <= stop + 1e-9:
        values.append(round(t, 4))
        t += step
    return values


def main():
    parser = argparse.ArgumentParser(description="Sweep thresholds for 3D fall checkpoint")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--features-dir", default=os.path.join("features", "3d"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated thresholds. Overrides --start/--stop/--step.",
    )
    parser.add_argument("--start", type=float, default=0.20)
    parser.add_argument("--stop", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    split_dir = os.path.join(args.features_dir, args.split)
    X = np.load(os.path.join(split_dir, "X.npy")).transpose(0, 2, 1)
    y = np.load(os.path.join(split_dir, "y.npy"))

    clf = FallClassifier(
        args.checkpoint,
        threshold_low=0.4,
        threshold_high=0.85,
        device=args.device,
    )
    p = clf.predict_proba(X)[:, 1]

    print(f"checkpoint={args.checkpoint}")
    print(f"split={args.split} samples={len(y)} fall_ratio={y.mean():.4f}")
    auc = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")
    print(f"AUC={auc:.4f}\n")

    for t in parse_thresholds(args.thresholds, args.start, args.stop, args.step):
        pred = (p >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y, pred, average=None, labels=[0, 1], zero_division=0
        )
        cm = confusion_matrix(y, pred, labels=[0, 1])
        print(
            f"threshold={t:.2f} | "
            f"normal: P={prec[0]:.3f} R={rec[0]:.3f} F1={f1[0]:.3f} | "
            f"fall: P={prec[1]:.3f} R={rec[1]:.3f} F1={f1[1]:.3f} | "
            f"CM={cm.tolist()}"
        )


if __name__ == "__main__":
    main()
