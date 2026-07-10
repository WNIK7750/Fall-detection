import torch
ckpt = torch.load('checkpoints/enhanced_run_20260508_124532/best_model.pth', map_location='cpu')
print("Keys in checkpoint:", list(ckpt.keys()))
if 'config' in ckpt:
    print("config keys:", list(ckpt['config'].keys()) if isinstance(ckpt['config'], dict) else 'not dict')
print("Has scaler_mean:", 'scaler_mean' in ckpt)
print("Has scaler_std:", 'scaler_std' in ckpt)
if 'scaler_mean' in ckpt:
    import numpy as np
    print("scaler_mean:", np.array(ckpt['scaler_mean']))
    print("scaler_std:", np.array(ckpt['scaler_std']))
