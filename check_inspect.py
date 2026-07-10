import torch
import glob
import os

os.chdir(r'D:\识别人类摔倒的智能系统')

runs = sorted(glob.glob('checkpoints/enhanced_run_*'))
print('All runs:')
for r in runs:
    print(r)

print()
print('=== Latest 3 runs config ===')
for r in runs[-3:]:
    print(f'\n=== {r} ===')
    ckpt_path = f'{r}/best_model.pth'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        config = ckpt.get('config', {})
        print('model_type:', config.get('model_type'))
        print('input_size:', config.get('input_size'))
        print('hidden_size:', config.get('hidden_size'))
        print('has scaler_mean:', 'scaler_mean' in config)
        print('has scaler_std:', 'scaler_std' in config)
        print('all config keys:', list(config.keys()))
    else:
        print('best_model.pth NOT FOUND')
