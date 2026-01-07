import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import cv2
import shutil
import random
from utils import Config, compute_metrics, get_param_norms, add_scale_bar, count_parameters
from models import UNet, WRTPNet

try:
    import bm3d
except Exception:
    bm3d = None

def register_result(method_name, idx, clean_np, pred_np, psnr, ssim, fsim, vif, msssim, fom, benchmark_results):
    if method_name not in benchmark_results:
        benchmark_results[method_name] = {'psnr': [], 'ssim': [], 'fsim': [], 'vif': [], 'msssim': [], 'fom': [], 'imgs': {}}
    benchmark_results[method_name]['psnr'].append(psnr)
    benchmark_results[method_name]['ssim'].append(ssim)
    benchmark_results[method_name]['fsim'].append(fsim)
    benchmark_results[method_name]['vif'].append(vif)
    benchmark_results[method_name]['msssim'].append(msssim)
    benchmark_results[method_name]['fom'].append(fom)
    
    if idx < Config.NUM_PLOT_SAMPLES:
        pred_with_bar = add_scale_bar(pred_np)
        benchmark_results[method_name]['imgs'][idx] = pred_with_bar
        try:
            outp = (pred_with_bar * 255).astype(np.uint8)[:,:,::-1]
            fname = os.path.join(Config.OUTPUT_DIR, f"{method_name}_img{idx}.png")
            cv2.imwrite(fname, outp)
        except Exception:
            pass

def geometric_ensemble_inference(model, x):
    dev = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else Config.DEVICE
    x = x.to(dev)
    model.eval()
    if len(x.shape) == 3: x = x.unsqueeze(0)
    outputs = []
    for k in [0, 1, 2, 3]:
        x_rot = torch.rot90(x, k, [2, 3])
        out_rot = model(x_rot)
        out_inv = torch.rot90(out_rot, -k, [2, 3])
        outputs.append(out_inv)
    x_flip = torch.flip(x, [3])
    for k in [0, 1, 2, 3]:
        x_rot = torch.rot90(x_flip, k, [2, 3])
        out_rot = model(x_rot)
        out_inv = torch.rot90(out_rot, -k, [2, 3])
        out_final = torch.flip(out_inv, [3])
        outputs.append(out_final)
    return torch.stack(outputs).mean(dim=0).clamp(0,1)

def mask_n2v(img, mask_ratio):
    b,c,h,w = img.shape
    mask = torch.rand((b,c,h,w), device=img.device) < mask_ratio
    offs = [(0,1),(0,-1),(1,0),(-1,0)]
    rep = img.clone()
    for dx,dy in random.sample(offs, k=4):
        shifted = torch.roll(img, shifts=(dx, dy), dims=(2,3))
        rep = torch.where(mask, shifted, rep)
    return rep, mask

def masked_l1_msssim_loss(pred, target, mask, alpha=0.84):
    mask_f = mask.float()
    l1 = F.l1_loss(pred * mask_f, target * mask_f, reduction='sum') / (mask_f.sum() + 1e-6)
    from utils import calc_msssim
    pred_visible = pred * mask_f + target * (1.0 - mask_f)
    ms_ssim_val = 1.0 - calc_msssim(pred_visible, target)
    return alpha * ms_ssim_val + (1.0 - alpha) * l1

def enable_dropout_only(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout) or isinstance(m, torch.nn.Dropout2d):
            m.train()

def run_traditional_methods(test_pairs, record_runtime_info, seed_dir, benchmark_results):
    marker = os.path.join(seed_dir, "done_Traditional.txt")
    if os.path.exists(marker):
        runt_path = os.path.join(seed_dir, "runtimes.json")
        if os.path.exists(runt_path):
            try:
                with open(runt_path, 'r') as f:
                    seed_runt = json.load(f)
                    for k in ['Gaussian','NLM','BM3D']:
                        if k in seed_runt: record_runtime_info[k] = seed_runt[k]
            except Exception: pass
        return

    print("  Running Traditional Methods...")
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    t_gauss, t_nlm, t_bm3d = [], [], []
    t0 = time.time()
    for i, (clean, noisy) in enumerate(test_pairs):
        c_np = clean.squeeze().cpu().permute(1,2,0).numpy()
        n_np = noisy.squeeze().cpu().permute(1,2,0).numpy()
        n_u8 = (np.clip(n_np, 0, 1) * 255).astype(np.uint8)

        st = time.time(); g = cv2.GaussianBlur(n_np, (5, 5), 0); t_gauss.append(time.time() - st)
        register_result('Gaussian', i, c_np, g, *compute_metrics(c_np, g), benchmark_results)

        st = time.time()
        nlm = cv2.fastNlMeansDenoisingColored(n_u8, None, 25, 25, 7, 21).astype(np.float32) / 255.0
        t_nlm.append(time.time() - st)
        register_result('NLM', i, c_np, nlm, *compute_metrics(c_np, nlm), benchmark_results)

        st = time.time()
        try:
            if bm3d is None: b = n_np
            else: b = np.clip(bm3d.bm3d(n_np.astype(np.float32), 25/255.0), 0, 1)
        except Exception: b = n_np
        t_bm3d.append(time.time() - st)
        register_result('BM3D', i, c_np, b, *compute_metrics(c_np, b), benchmark_results)

    for k, v in zip(['Gaussian', 'NLM', 'BM3D'], [t_gauss, t_nlm, t_bm3d]):
        avg = float(np.mean(v)) if v else 0.0
        record_runtime_info[k] = {
            'train_time': 0.0, 'inf_time_avg': avg, 'params': 0,
            'inf_time_per_256_ms': avg * 1000.0,
            'gpu_mem_mb': float(torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0
        }
    open(marker, 'w').close()

def run_noise2void(test_pairs, train_loader, train_on_raw=False, benchmark_results=None, NoiseInjector=None):
    from utils import get_param_norms
    model = UNet(base=64, dropout=0.0).to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    p_norms, loss_hist = [], []
    t_start = time.time()
    for epoch in range(Config.EPOCHS):
        model.train(); e_loss = 0.0
        for clean in train_loader:
            img = clean.to(Config.DEVICE)
            noisy = img.clone() if train_on_raw else NoiseInjector.add_gaussian(img, sigma=Config.SIGMA)
            m_inp, mask = mask_n2v(noisy, Config.MASK_PERCENT)
            optimizer.zero_grad()
            pred = model(m_inp)
            loss = masked_l1_msssim_loss(pred, noisy, mask)
            loss.backward(); optimizer.step()
            e_loss += float(loss.item())
        p_norms.append(get_param_norms(model)); loss_hist.append(e_loss)
    t_train = time.time() - t_start
    torch.save(model.state_dict(), os.path.join(Config.OUTPUT_DIR, "tmp_model_state.pth"))
    
    inf_times = []
    model.eval()
    with torch.no_grad():
        for i, (clean, noisy) in enumerate(test_pairs):
            inp = noisy if not train_on_raw else clean
            st = time.time(); out = geometric_ensemble_inference(model, inp); inf_times.append(time.time() - st)
            c_np, o_np = clean.squeeze().cpu().permute(1,2,0).numpy(), out.squeeze().cpu().permute(1,2,0).numpy()
            register_result('N2V' + ('_raw' if train_on_raw else ''), i, c_np, o_np, *compute_metrics(c_np, o_np), benchmark_results)
    return t_train, np.mean(inf_times), count_parameters(model), p_norms, loss_hist

def run_ne2ne(test_pairs, train_loader, train_on_raw=False, benchmark_results=None, NoiseInjector=None):
    model = UNet(base=64, dropout=0.0).to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    p_norms, loss_hist = [], []
    def sub_sample(img): return [img[:,:,0::2,0::2], img[:,:,0::2,1::2], img[:,:,1::2,0::2], img[:,:,1::2,1::2]]
    t_start = time.time()
    for epoch in range(Config.EPOCHS):
        model.train(); e_loss = 0.0
        for clean in train_loader:
            img = clean.to(Config.DEVICE)
            noisy = img.clone() if train_on_raw else NoiseInjector.add_gaussian(img, sigma=Config.SIGMA)
            subs = sub_sample(noisy); idx = np.random.choice(4, 2, replace=False)
            optimizer.zero_grad(); out = model(subs[idx[0]])
            loss = F.l1_loss(out, subs[idx[1]]); loss.backward(); optimizer.step()
            e_loss += float(loss.item())
        p_norms.append(get_param_norms(model)); loss_hist.append(e_loss)
    t_train = time.time() - t_start
    torch.save(model.state_dict(), os.path.join(Config.OUTPUT_DIR, "tmp_model_state.pth"))

    inf_times = []
    model.eval()
    with torch.no_grad():
        for i, (clean, noisy) in enumerate(test_pairs):
            inp = noisy if not train_on_raw else clean
            subs = sub_sample(inp); st = time.time(); preds = [model(s) for s in subs]
            B,C,Hs,Ws = preds[0].shape; full = torch.zeros((B,C,Hs*2,Ws*2), device=inp.device)
            positions = [(0,0),(0,1),(1,0),(1,1)]
            for p_idx, p in enumerate(preds): r,cp = positions[p_idx]; full[:,:,r::2,cp::2] = p
            inf_times.append(time.time() - st)
            c_np, o_np = clean.squeeze().cpu().permute(1,2,0).numpy(), full.squeeze().cpu().permute(1,2,0).numpy()
            register_result('Ne2Ne' + ('_raw' if train_on_raw else ''), i, c_np, o_np, *compute_metrics(c_np, o_np), benchmark_results)
    return t_train, np.mean(inf_times), count_parameters(model), p_norms, loss_hist

def run_self2self(test_pairs, train_loader, train_on_raw=False, benchmark_results=None, NoiseInjector=None):
    model = UNet(base=64, dropout=0.3).to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    p_norms, loss_hist = [], []
    t_start = time.time()
    for epoch in range(Config.EPOCHS):
        model.train(); e_loss = 0.0
        for clean in train_loader:
            img = clean.to(Config.DEVICE)
            noisy = img.clone() if train_on_raw else NoiseInjector.add_gaussian(img, sigma=Config.SIGMA)
            m_inp, mask = mask_n2v(noisy, Config.MASK_PERCENT)
            optimizer.zero_grad(); pred = model(m_inp)
            loss = masked_l1_msssim_loss(pred, noisy, mask)
            loss.backward(); optimizer.step()
            e_loss += float(loss.item())
        p_norms.append(get_param_norms(model)); loss_hist.append(e_loss)
    t_train = time.time() - t_start
    torch.save(model.state_dict(), os.path.join(Config.OUTPUT_DIR, "tmp_model_state.pth"))

    enable_dropout_only(model)
    inf_times = []
    with torch.no_grad():
        for i, (clean, noisy) in enumerate(test_pairs):
            inp = noisy if not train_on_raw else clean
            preds = []; st = time.time()
            for _ in range(10): preds.append(model(inp))
            avg_pred = torch.stack(preds).mean(dim=0).clamp(0,1); inf_times.append(time.time() - st)
            c_np, o_np = clean.squeeze().cpu().permute(1,2,0).numpy(), avg_pred.squeeze().cpu().permute(1,2,0).numpy()
            register_result('Self2Self' + ('_raw' if train_on_raw else ''), i, c_np, o_np, *compute_metrics(c_np, o_np), benchmark_results)
    return t_train, np.mean(inf_times), count_parameters(model), p_norms, loss_hist

def run_wrtpnet(test_pairs, train_loader, train_on_raw=False, benchmark_results=None, NoiseInjector=None):
    model = WRTPNet(base=96, num_blocks=6).to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    p_norms, loss_hist = [], []
    t_start = time.time()
    for epoch in range(Config.EPOCHS):
        model.train(); e_loss = 0.0
        for clean in train_loader:
            img = clean.to(Config.DEVICE)
            noisy = img.clone() if train_on_raw else NoiseInjector.add_gaussian(img, sigma=Config.SIGMA)
            m_inp, mask = mask_n2v(noisy, Config.MASK_PERCENT)
            optimizer.zero_grad(); out = model(m_inp)
            loss = masked_l1_msssim_loss(out, noisy, mask)
            loss.backward(); optimizer.step()
            e_loss += float(loss.item())
        p_norms.append(get_param_norms(model)); loss_hist.append(e_loss)
    t_train = time.time() - t_start
    torch.save(model.state_dict(), os.path.join(Config.OUTPUT_DIR, "tmp_model_state.pth"))

    model.eval()
    inf_times = []
    with torch.no_grad():
        for i, (clean, noisy) in enumerate(test_pairs):
            inp = noisy if not train_on_raw else clean
            st = time.time(); out = geometric_ensemble_inference(model, inp); inf_times.append(time.time() - st)
            c_np, o_np = clean.squeeze().cpu().permute(1,2,0).numpy(), out.squeeze().cpu().permute(1,2,0).numpy()
            register_result('WRTPNet' + ('_raw' if train_on_raw else ''), i, c_np, o_np, *compute_metrics(c_np, o_np), benchmark_results)
    return t_train, np.mean(inf_times), count_parameters(model), p_norms, loss_hist
