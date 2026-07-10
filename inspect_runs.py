import torch
import glob
import os

runs = sorted(glob.glob('checkpoints/enhanced_run_*'))

print("=" * 60)
print("所有训练 Run 的模型配置")
print("=" * 60)

for r in runs:
    print(f"\n【{os.path.basename(r)}】")
    ckpt_path = f'{r}/best_model.pth'
    if not os.path.exists(ckpt_path):
        print("  (文件不存在)")
        continue

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})

    print(f"  model_type: {cfg.get('model_type', 'N/A')}")
    print(f"  input_size: {cfg.get('input_size', 'N/A')}")
    print(f"  hidden_size: {cfg.get('hidden_size', 'N/A')}")
    print(f"  has scaler_mean: {'scaler_mean' in cfg}")
    print(f"  has scaler_std: {'scaler_std' in cfg}")

    metrics = ckpt.get('metrics', {})
    if metrics:
        print(f"  val_f1: {metrics.get('val_f1', 'N/A')}")
        print(f"  val_recall: {metrics.get('val_recall', 'N/A')}")
