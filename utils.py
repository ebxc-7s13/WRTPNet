import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from scipy.ndimage import distance_transform_edt

class Config:
    INPUT_DIR = " input images path" #image dataset path
    OUTPUT_DIR = "results" #results saving folder path
    IMG_SIZE = 256
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    MASK_PERCENT = 0.03
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4 if torch.cuda.is_available() else 0
    PIN_MEMORY = True if torch.cuda.is_available() else False
    NUM_PLOT_SAMPLES = 5
    SEEDS = [42, 100, 123]
    PIXEL_SIZE_UM = 1.0
    SIGMA = 0.0980
    POISSON_PEAK = 30.0
    SIGMA_READ = 0.02
    AGGREGATED_DIR = os.path.join(OUTPUT_DIR, "aggregated")
    ZIP_OUTPUT = os.path.join(OUTPUT_DIR, "results_all.zip")
    MASTER_ZIP_FINAL = os.path.join(OUTPUT_DIR, "seeds_master_collection.zip")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def cleanup():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_param_norms(model):
    norms = []
    for p in model.parameters():
        if p.requires_grad:
            norms.append(p.detach().cpu().float().norm().item())
    return float(np.mean(norms)) if norms else 0.0

def add_scale_bar(img_np, pixel_size_um=Config.PIXEL_SIZE_UM):
    vis = (np.clip(img_np, 0, 1) * 255).astype(np.uint8).copy()
    h, w, _ = vis.shape
    bar_len_px = min(50, max(1, w - 60))
    bar_len_um = bar_len_px * pixel_size_um
    start_x = max(10, w - bar_len_px - 20)
    cv2.line(vis, (start_x, h - 20), (start_x + bar_len_px, h - 20), (255, 255, 255), 2)
    label = f"{bar_len_um:.0f} um"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(label, font, 0.5, 1)
    cv2.putText(vis, label, (start_x + (bar_len_px // 2) - (text_w // 2), h - 30), font, 0.5, (255, 255, 255), 1)
    return vis.astype(np.float32) / 255.0

def calc_msssim(img1_tensor, img2_tensor, window_size=11, val_range=1.0):
    if img1_tensor.ndim == 3: img1_tensor = img1_tensor.unsqueeze(0)
    if img2_tensor.ndim == 3: img2_tensor = img2_tensor.unsqueeze(0)
    pad = window_size // 2
    def _gaussian(w_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
        return gauss/gauss.sum()
    def _create_window(w_size, channel):
        _1D_window = _gaussian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(channel, 1, w_size, w_size).to(img1_tensor.device)
    window = _create_window(window_size, img1_tensor.shape[1])
    def _ssim(img1, img2, window, val_range):
        mu1 = F.conv2d(img1, window, padding=pad, groups=img1.shape[1])
        mu2 = F.conv2d(img2, window, padding=pad, groups=img1.shape[1])
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1*mu2
        sigma1_sq = F.conv2d(img1*img1, window, padding=pad, groups=img1.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=pad, groups=img1.shape[1]) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=pad, groups=img1.shape[1]) - mu1_mu2
        C1, C2 = (0.01 * val_range) ** 2, (0.03 * val_range) ** 2
        luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        contrast = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        return luminance, contrast
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(img1_tensor.device)
    mcs = []
    for i in range(5):
        lum, con = _ssim(img1_tensor, img2_tensor, window, val_range)
        mcs.append(torch.relu(con).mean(dim=(1, 2, 3)))
        if i < 4:
            img1_tensor, img2_tensor = F.avg_pool2d(img1_tensor, (2, 2)), F.avg_pool2d(img2_tensor, (2, 2))
    pow1 = torch.stack(mcs) ** weights.view(-1, 1)
    return torch.prod(pow1, dim=0).mean().item()

def calc_fsim(clean, pred):
    Y_c = 0.299 * clean[:,:,0] + 0.587 * clean[:,:,1] + 0.114 * clean[:,:,2]
    Y_p = 0.299 * pred[:,:,0] + 0.587 * pred[:,:,1] + 0.114 * pred[:,:,2]
    Y_c, Y_p = Y_c.astype(np.float32), Y_p.astype(np.float32)
    GM_c = np.sqrt(cv2.Scharr(Y_c, cv2.CV_32F, 1, 0)**2 + cv2.Scharr(Y_c, cv2.CV_32F, 0, 1)**2)
    GM_p = np.sqrt(cv2.Scharr(Y_p, cv2.CV_32F, 1, 0)**2 + cv2.Scharr(Y_p, cv2.CV_32F, 0, 1)**2)
    S_GM = (2 * GM_c * GM_p + 0.0026) / (GM_c**2 + GM_p**2 + 0.0026)
    S_L = (2 * Y_c * Y_p + 0.0026) / (Y_c**2 + Y_p**2 + 0.0026)
    Wm = np.maximum(GM_c, GM_p)
    return np.sum(S_GM * S_L * Wm) / (np.sum(Wm) + 1e-8)

def calc_vif(clean, pred):
    clean, pred = clean.astype(np.float32), pred.astype(np.float32)
    num, den = 0.0, 0.0
    for scale in range(1, 5):
        N = 2**(4-scale+1) + 1
        mu1, mu2 = cv2.GaussianBlur(clean, (N, N), N/3.0), cv2.GaussianBlur(pred, (N, N), N/3.0)
        sigma1_sq = cv2.GaussianBlur(clean * clean, (N, N), N/3.0) - mu1**2
        sigma2_sq = cv2.GaussianBlur(pred * pred, (N, N), N/3.0) - mu2**2
        sigma12 = cv2.GaussianBlur(clean * pred, (N, N), N/3.0) - mu1*mu2
        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g * sigma12
        num += np.sum(np.log10(1 + g**2 * sigma1_sq / (sv_sq + 2.0)))
        den += np.sum(np.log10(1 + sigma1_sq / 2.0))
        if scale < 4:
            clean, pred = cv2.resize(clean, (0,0), fx=0.5, fy=0.5), cv2.resize(pred, (0,0), fx=0.5, fy=0.5)
    return num / (den + 1e-8)

def calc_fom(clean, pred):
    c_gray = cv2.cvtColor((clean*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    p_gray = cv2.cvtColor((pred*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges_c, edges_p = cv2.Canny(c_gray, 50, 150), cv2.Canny(p_gray, 50, 150)
    if np.sum(edges_c) == 0: return 1.0
    dist_map = distance_transform_edt(255 - edges_c)
    N_c, N_p = np.sum(edges_c > 0), np.sum(edges_p > 0)
    if max(N_c, N_p) == 0: return 0.0
    return np.sum(1.0 / (1.0 + (1.0/9.0) * (dist_map[edges_p > 0] ** 2))) / max(N_c, N_p)

def compute_metrics(clean, pred):
    p = psnr_metric(clean, pred, data_range=1.0)
    s = ssim_metric(clean, pred, data_range=1.0, channel_axis=2)
    f = calc_fsim(clean, pred)
    v = calc_vif(clean, pred)
    try:
        t_c = torch.from_numpy(clean.transpose(2,0,1)).float().to(Config.DEVICE)
        t_p = torch.from_numpy(pred.transpose(2,0,1)).float().to(Config.DEVICE)
        m = calc_msssim(t_c, t_p)
    except: m = np.nan
    fo = calc_fom(clean, pred)
    return p, s, f, v, m, fo
