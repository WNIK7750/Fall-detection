"""
Train 3-class fall classifier on MCFD 3D features.

Uses BiLSTM+BiGRU+Attention architecture with Focal Loss + SMOTE balancing.
Outputs binary model; 3-class decisions via dual thresholds at inference time.

Usage:
    python scripts/training/train_3d.py                          # full training
    python scripts/training/train_3d.py --epochs 50 --lr 0.0005  # custom params
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from models.enhanced_lstm import EnhancedFallDetector
from pipeline.classifier import FallClassifier

EXPECTED_SEQUENCE_LENGTH = 40


# ---------------------------------------------------------------------------
# Focal Loss (binary)
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 2.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.CrossEntropyLoss(reduction="none")(logits, targets)
        pt = torch.exp(-ce)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * ce).mean()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 20260713):
    """Fix random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args):
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Random seed fixed: {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {vars(args)}")

    # Load data
    feats_dir = REPO_ROOT / "features" / "3d"
    X_train = np.load(feats_dir / "train" / "X.npy")
    y_train = np.load(feats_dir / "train" / "y.npy")
    X_val = np.load(feats_dir / "val" / "X.npy")
    y_val = np.load(feats_dir / "val" / "y.npy")

    N, F, T = X_train.shape
    if T != EXPECTED_SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected {EXPECTED_SEQUENCE_LENGTH}-frame training windows, got {T}. "
            "Regenerate features with scripts/training/extract_dataset_3d.py."
        )
    if X_val.ndim != 3 or X_val.shape[1:] != (F, T):
        raise ValueError(
            f"Validation shape {X_val.shape} does not match training feature shape (*, {F}, {T})."
        )
    print(f"\nData: train={N} val={len(y_val)}, shape=({F},{T}), "
          f"fall_ratio={y_train.mean():.1%}")
    print(f"  Train class dist: 0={int((y_train==0).sum())}  1={int((y_train==1).sum())}")
    print(f"  Val   class dist: 0={int((y_val==0).sum())}  1={int((y_val==1).sum())}")

    if y_train.mean() < 0.01:
        print("  WARNING: train set has <1% fall samples — model will struggle to learn!")
    if y_val.mean() < 0.01:
        print("  WARNING: val set has <1% fall samples — cannot evaluate properly!")

    # Standardize (fit on train, apply to val)
    X_train_2d = X_train.transpose(0, 2, 1).reshape(-1, F)  # (N*T, F)
    X_val_2d = X_val.transpose(0, 2, 1).reshape(-1, F)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(N, T, F)
    X_val_scaled = scaler.transform(X_val_2d).reshape(len(y_val), T, F)

    # Save scaler for inference
    scaler_dir = feats_dir / "scaler"
    scaler_dir.mkdir(parents=True, exist_ok=True)
    np.save(scaler_dir / "mean.npy", scaler.mean_)
    np.save(scaler_dir / "scale.npy", scaler.scale_)

    # SMOTE oversampling (flatten windows for SMOTE)
    if args.smote:
        X_flat = X_train_scaled.reshape(N, -1)
        sm = SMOTE(random_state=args.seed if args.seed is not None else 20260713)
        X_res, y_res = sm.fit_resample(X_flat, y_train)
        X_train_scaled = X_res.reshape(-1, T, F)
        y_train = y_res
        print(f"After SMOTE: train={len(y_train)}, fall_ratio={y_train.mean():.1%}")

    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model
    model = EnhancedFallDetector(
        input_size=F,
        hidden_size=128,
        num_classes=2,
        dropout=args.dropout,
    ).to(device)

    # Mixed precision
    scaler_amp = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Optimizer & loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = FocalLoss(alpha=2.0, gamma=2.0)

    # Training loop — monitor val_loss (stable, not noisy like AUC on small val set)
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": []}
    patience_counter = 0
    run_name = datetime.now().strftime("enhanced_3d_%Y%m%d_%H%M%S")
    ckpt_dir = REPO_ROOT / "checkpoints" / run_name  # always set, even if no improvement
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            if scaler_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            train_loss += loss.item()

        history["train_loss"].append(train_loss / len(train_loader))

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item()
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(yb.cpu().numpy())

        history["val_loss"].append(val_loss / len(val_loader))
        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        # Use probability threshold (not argmax) — same as inference
        preds = (probs[:, 1] >= args.threshold).astype(int)
        f1 = classification_report(labels, preds, output_dict=True, zero_division=0)["1"]["f1-score"]
        auc = roc_auc_score(labels, probs[:, 1]) if len(set(labels)) > 1 else 0.5
        history["val_f1"].append(f1)
        history["val_auc"].append(auc)

        val_loss_epoch = history["val_loss"][-1]
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d}/{args.epochs} | loss={train_loss/len(train_loader):.4f} "
              f"val_loss={val_loss_epoch:.4f} | F1={f1:.3f} AUC={auc:.3f} | lr={lr_now:.2e}")

        # Save best by val_loss (more stable than AUC/F1 on small val set)
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "input_size": F,
                "sequence_length": T,
                "epoch": epoch,
                "val_loss": val_loss_epoch,
                "f1": f1,
                "auc": auc,
            }, ckpt_dir / "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save history (always, even if model never improved)
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save last checkpoint if no best was saved (model never improved)
    best_path = ckpt_dir / "best_model.pth"
    if not best_path.exists():
        print("No improvement during training — saving last state")
        torch.save({
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "input_size": F,
            "sequence_length": T,
            "epoch": epoch,
            "val_loss": val_loss_epoch,
            "f1": f1,
            "auc": auc,
        }, best_path)

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()
    print(
        f"Reloaded best checkpoint from epoch {best_checkpoint.get('epoch')} "
        f"for final test evaluation"
    )

    # Test set evaluation
    test_path = feats_dir / "test" / "X.npy"
    if test_path.exists() and test_path.stat().st_size > 128:
        X_test = np.load(test_path)
        y_test = np.load(feats_dir / "test" / "y.npy")
        if X_test.ndim != 3 or X_test.shape[1:] != (F, T):
            raise ValueError(
                f"Test shape {X_test.shape} does not match training feature shape (*, {F}, {T})."
            )
        print(f"\n=== Test Set ({run_name}) ===")
        print(f"  Samples: {len(y_test)}, fall ratio: {y_test.mean():.1%}")

        if len(y_test) == 0:
            print("  (Test set empty, skipping evaluation)")
        elif len(set(y_test)) < 2:
            print(f"  WARNING: test set has only 1 class ({int(y_test[0])}). "
                  f"Cannot compute AUC/F1 meaningfully.")
            # Still run inference for completeness
            X_test_2d = X_test.transpose(0, 2, 1).reshape(-1, F)
            X_test_scaled = scaler.transform(X_test_2d).reshape(len(y_test), T, F)
            model.eval()
            with torch.no_grad():
                x_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
                logits = model(x_t)
                t_probs = torch.softmax(logits, dim=1).cpu().numpy()
            t_preds = (t_probs[:, 1] >= args.threshold).astype(int)
            acc = (t_preds == y_test).mean()
            print(f"  Accuracy: {acc:.3f}  (all samples are class {int(y_test[0])})")
        else:
            X_test_2d = X_test.transpose(0, 2, 1).reshape(-1, F)
            X_test_scaled = scaler.transform(X_test_2d).reshape(len(y_test), T, F)
            model.eval()
            with torch.no_grad():
                x_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
                logits = model(x_t)
                t_probs = torch.softmax(logits, dim=1).cpu().numpy()
            t_preds = (t_probs[:, 1] >= args.threshold).astype(int)

            present_labels = sorted(set(int(y) for y in y_test) | set(int(p) for p in t_preds))
            label_names = ["normal", "fall"]
            used_names = [label_names[i] for i in present_labels if i < len(label_names)]
            print(classification_report(y_test, t_preds, labels=present_labels,
                                        target_names=used_names, zero_division=0))
            print(f"AUC-ROC: {roc_auc_score(y_test, t_probs[:,1]):.4f}")
            cm = confusion_matrix(y_test, t_preds)
            print(f"Confusion Matrix:\n{cm}")

    print(f"\nCheckpoint saved: {ckpt_dir}/best_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility; omit for default randomness")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Fall probability threshold for classification")
    parser.add_argument("--smote", action="store_true", default=True)
    parser.add_argument("--no-smote", action="store_false", dest="smote")
    args = parser.parse_args()
    train(args)
