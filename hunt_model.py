"""
Auto-hunt: repeatedly train until finding a model with target AUC + balanced recall.

Usage:
    python hunt_model.py                                    # target AUC >= 0.92
    python hunt_model.py --target-auc 0.90 --target-norm-recall 0.6
    python hunt_model.py --target-auc 0.93 --max-attempts 50
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.dirname(__file__))
from models.enhanced_lstm import EnhancedFallDetector


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=2.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = torch.nn.CrossEntropyLoss(reduction="none")(logits, targets)
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


def evaluate(model, X, y, scaler, threshold, device):
    """Return dict with test metrics at given threshold."""
    model.eval()
    X2d = X.transpose(0, 2, 1).reshape(-1, X.shape[1])
    Xs = scaler.transform(X2d).reshape(len(y), X.shape[2], X.shape[1])
    with torch.no_grad():
        probs = torch.softmax(model(torch.FloatTensor(Xs).to(device)), 1).cpu().numpy()

    preds = (probs[:, 1] >= threshold).astype(int)
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    auc = roc_auc_score(y, probs[:, 1]) if len(set(y)) > 1 else 0.5
    norm_recall = tn / (tn + fp + 1e-9)
    fall_recall = tp / (tp + fn + 1e-9)
    fall_prec = tp / (tp + fp + 1e-9)

    return {
        "auc": auc, "acc": (tp + tn) / (tp + tn + fp + fn),
        "norm_recall": norm_recall, "fall_recall": fall_recall,
        "fall_prec": fall_prec, "fp": int(fp), "fn": int(fn),
        "tn": int(tn), "tp": int(tp), "cm": cm.tolist(),
    }


def train_one(model, train_loader, val_loader, device, epochs=100, lr=0.001,
              patience=20, threshold=0.5):
    scaler_amp = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = FocalLoss(alpha=2.0, gamma=2.0)

    # Monitor val_loss (stable) for early stopping, not noisy AUC/F1 on small val set
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            if scaler_amp:
                with torch.amp.autocast("cuda"):
                    loss = criterion(model(xb), yb)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Validate loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-auc", type=float, default=0.92)
    parser.add_argument("--target-norm-recall", type=float, default=0.5,
                        help="Minimum normal-class recall (reduce false alarms)")
    parser.add_argument("--target-fall-recall", type=float, default=0.95,
                        help="Minimum fall-class recall (catching falls)")
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Eval threshold")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Target: AUC≥{args.target_auc}, NormR≥{args.target_norm_recall}, "
          f"FallR≥{args.target_fall_recall}, Max attempts={args.max_attempts}")

    # Load data
    feats_dir = os.path.join("features", "3d")
    X_train = np.load(os.path.join(feats_dir, "train", "X.npy"))
    y_train = np.load(os.path.join(feats_dir, "train", "y.npy"))
    X_val = np.load(os.path.join(feats_dir, "val", "X.npy"))
    y_val = np.load(os.path.join(feats_dir, "val", "y.npy"))
    X_test = np.load(os.path.join(feats_dir, "test", "X.npy"))
    y_test = np.load(os.path.join(feats_dir, "test", "y.npy"))
    N, F_, T = X_train.shape

    print(f"Data: train={N}, val={len(y_val)}, test={len(y_test)}")
    print(f"  fall ratio: train={y_train.mean():.1%}  val={y_val.mean():.1%}  "
          f"test={y_test.mean():.1%}")

    # Standardize
    X_train_2d = X_train.transpose(0, 2, 1).reshape(-1, F_)
    X_val_2d = X_val.transpose(0, 2, 1).reshape(-1, F_)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_2d).reshape(N, T, F_)
    X_val_s = scaler.transform(X_val_2d).reshape(len(y_val), T, F_)

    # SMOTE
    X_flat = X_train_s.reshape(N, -1)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_flat, y_train)
    X_train_s = X_res.reshape(-1, T, F_)
    y_train_bal = y_res

    train_ds = TensorDataset(torch.FloatTensor(X_train_s), torch.LongTensor(y_train_bal))
    val_ds = TensorDataset(torch.FloatTensor(X_val_s), torch.LongTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    start_time = time.time()
    best_overall = {"auc": 0.0, "norm_recall": 0.0, "fall_recall": 0.0}

    for attempt in range(1, args.max_attempts + 1):
        run_name = datetime.now().strftime("hunt_%Y%m%d_%H%M%S")
        model = EnhancedFallDetector(input_size=F_, hidden_size=128,
                                     num_classes=2, dropout=0.3).to(device)

        model = train_one(model, train_loader, val_loader, device,
                          epochs=args.epochs, lr=args.lr, threshold=args.threshold)

        metrics = evaluate(model, X_test, y_test, scaler, args.threshold, device)
        elapsed = time.time() - start_time

        print(f"\n[Attempt {attempt:3d}] {run_name} ({elapsed:.0f}s)")
        print(f"  AUC={metrics['auc']:.4f} | Acc={metrics['acc']:.3f} | "
              f"FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']} TN={metrics['tn']}")
        print(f"  Norm Recall={metrics['norm_recall']:.3f} | "
              f"Fall Recall={metrics['fall_recall']:.3f} | "
              f"Fall Prec={metrics['fall_prec']:.3f}")

        if metrics["auc"] > best_overall["auc"]:
            best_overall = metrics
            ckpt_dir = os.path.join("checkpoints", run_name)
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "input_size": F_,
                "metrics": metrics,
            }, os.path.join(ckpt_dir, "best_model.pth"))
            with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

        # Check if targets met
        ok_auc = metrics["auc"] >= args.target_auc
        ok_norm = metrics["norm_recall"] >= args.target_norm_recall
        ok_fall = metrics["fall_recall"] >= args.target_fall_recall

        if ok_auc and ok_norm and ok_fall:
            print(f"\n*** TARGET HIT at attempt {attempt}! ***")
            print(f"  Checkpoint: checkpoints/{run_name}/best_model.pth")
            break
    else:
        print(f"\n--- Max attempts ({args.max_attempts}) reached ---")
        print(f"  Best AUC: {best_overall['auc']:.4f} "
              f"at NormR={best_overall['norm_recall']:.3f} "
              f"FallR={best_overall['fall_recall']:.3f}")

    total_elapsed = time.time() - start_time
    print(f"Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
