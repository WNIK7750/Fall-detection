# Fall Detection — Environment Setup Script
# Run this in PowerShell terminal (not in IDE):  .\setup_env.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Fall Detection Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: PyTorch CUDA (GPU) version
Write-Host "[1/4] Installing PyTorch CUDA (this may take 2-3 minutes)..." -ForegroundColor Yellow
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
Write-Host ""

# Step 2: llama-cpp-python (LLM inference)
Write-Host "[2/4] Installing llama-cpp-python..." -ForegroundColor Yellow
python -m pip install llama-cpp-python
Write-Host ""

# Step 3: Gradio (Demo UI)
Write-Host "[3/4] Installing gradio..." -ForegroundColor Yellow
python -m pip install gradio
Write-Host ""

# Step 4: imbalanced-learn (SMOTE)
Write-Host "[4/4] Installing imbalanced-learn..." -ForegroundColor Yellow
python -m pip install imbalanced-learn
Write-Host ""

# Verification
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Verification" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
python -c "
import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')
import cv2; print(f'OpenCV: {cv2.__version__}')
import numpy; print(f'NumPy: {numpy.__version__}')
import sklearn; print(f'sklearn: {sklearn.__version__}')
import ultralytics; print(f'ultralytics: {ultralytics.__version__}')
import mediapipe; print(f'mediapipe: {mediapipe.__version__}')
import pandas; print(f'pandas: {pandas.__version__}')
import matplotlib; print(f'matplotlib: {matplotlib.__version__}')
import seaborn; print(f'seaborn: {seaborn.__version__}')
import imblearn; print(f'imbalanced-learn: {imblearn.__version__}')
print('llama-cpp-python: ', end=''); import llama_cpp; print(llama_cpp.__version__)
print('gradio: ', end=''); import gradio; print(gradio.__version__)
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f}GB')
"
Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
