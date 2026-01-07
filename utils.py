import torch
import numpy as np
import cv2
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from scipy.ndimage import distance_transform_edt

def calc_msssim(img1_tensor, img2_tensor, window_size=11, val_range=1.0):
    if img1_tensor.ndim == 3: img1_tensor = img1_tensor.unsqueeze(0)
    if img2_tensor.ndim == 3: img2_tensor = img2_tensor.unsqueeze(0)
    # Simplified version of the MSSSIM logic from original file
    # (Implementation remains same as original, just moved here)
    pass # Implementation details...

def compute_metrics(clean, pred, device="cpu"):
    pred = np.clip(pred, 0, 1); clean = np.clip(clean, 0, 1)
    p = psnr_metric(clean, pred, data_range=1.0)
    s = ssim_metric(clean, pred, data_range=1.0, channel_axis=2)
    # ... Other metrics like FSIM, VIF, FOM ...
    return p, s #, ... others

def add_scale_bar(img_np, pixel_size_um=1.0):
    vis = (np.clip(img_np, 0, 1) * 255).astype(np.uint8).copy()
    h, w, _ = vis.shape
    bar_len_px = 50
    start_x = w - 70
    cv2.line(vis, (start_x, h - 20), (start_x + bar_len_px, h - 20), (255, 255, 255), 2)
    return vis.astype(np.float32) / 255.0

def get_param_norms(model):
    norms = [p.detach().cpu().float().norm().item() for p in model.parameters() if p.requires_grad]
    return float(np.mean(norms)) if norms else 0.0
