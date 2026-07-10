import torch

ckpt = torch.load(
    r'D:\识别人类摔倒的智能系统\checkpoints\enhanced_run_20260508_134356\best_model.pth',
    map_location='cpu',
    weights_only=False
)

print('=== Keys ===')
print(list(ckpt.keys()))
print()
print('=== Config ===')
for k, v in ckpt.get('config', {}).items():
    print(f'{k}: {v}')
print()
if 'metrics' in ckpt:
    print('=== Metrics ===')
    for k, v in ckpt['metrics'].items():
        print(f'{k}: {v}')
