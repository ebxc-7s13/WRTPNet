import os
import glob
import json
import time
import zipfile
import shutil
import torch
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils import Config, seed_everything, cleanup, add_scale_bar, compute_metrics, get_param_norms
from dataset import BenchmarkDataset, NoiseInjector
from engine import run_traditional_methods, run_noise2void, run_ne2ne, run_self2self, run_wrtpnet

def register_result(method_name, idx, clean_np, pred_np, psnr, ssim, fsim, vif, msssim, fom, benchmark_results):
    if method_name not in benchmark_results:
        benchmark_results[method_name] = {'psnr': [], 'ssim': [], 'fsim': [], 'vif': [], 'msssim': [], 'fom': [], 'imgs': {}}
    res = benchmark_results[method_name]
    for k, v in zip(['psnr','ssim','fsim','vif','msssim','fom'], [psnr, ssim, fsim, vif, msssim, fom]):
        res[k].append(v)
    if idx < Config.NUM_PLOT_SAMPLES:
        res['imgs'][idx] = add_scale_bar(pred_np)
        try:
            outp = (add_scale_bar(pred_np) * 255).astype(np.uint8)[:,:,::-1]
            fname = os.path.join(Config.OUTPUT_DIR, f"{method_name}_img{idx}.png")
            import cv2
            cv2.imwrite(fname, outp)
        except Exception:
            pass

if __name__ == "__main__":
    master_results = {}
    NOISE_TYPES = ["Gaussian", "Poisson", "PoissonGauss"]
    per_image_store = {nt: {} for nt in NOISE_TYPES}
    per_seed_param_norms = {nt: {} for nt in NOISE_TYPES}
    per_seed_losses = {nt: {} for nt in NOISE_TYPES}
    runtimes_log = {nt: {} for nt in NOISE_TYPES}
    last_benchmark_for_plot = None

    dataset = BenchmarkDataset(Config.INPUT_DIR)
    n = len(dataset)
    if n == 0: raise ValueError("No data found")
    train_n = int(0.8 * n)
    train_ds, test_ds = torch.utils.data.Subset(dataset, range(train_n)), torch.utils.data.Subset(dataset, range(train_n, n))
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)

    for noise_type in NOISE_TYPES:
        print(f"\n>>>> STARTING NOISE MODE: {noise_type} <<<<")
        master_results[noise_type] = {}
        for seed in Config.SEEDS:
            print(f"  >> SEED: {seed}")
            seed_everything(seed)
            benchmark_results = {}
            runtimes_log[noise_type].setdefault(seed, {})
            test_pairs = []
            seed_dir = os.path.join(Config.AGGREGATED_DIR, noise_type, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)

            # Build test pairs
            for i, clean in enumerate(test_loader):
                if i >= 50: break
                clean = clean.to(Config.DEVICE)
                if noise_type == "Gaussian": noisy = NoiseInjector.add_gaussian(clean, sigma=Config.SIGMA)
                elif noise_type == "Poisson": noisy = NoiseInjector.add_poisson(clean, peak=Config.POISSON_PEAK)
                elif noise_type == "PoissonGauss": noisy = NoiseInjector.add_poisson_gaussian(clean, peak=Config.POISSON_PEAK, sigma_read=Config.SIGMA_READ)
                test_pairs.append((clean, noisy))

            # Register Clean and Noisy
            for special in ['Clean', 'Noisy']:
                benchmark_results[special] = {'psnr': [], 'ssim': [], 'fsim': [], 'vif': [], 'msssim': [], 'fom': [], 'imgs': {}}

            for idx in range(min(Config.NUM_PLOT_SAMPLES, len(test_pairs))):
                c_t, n_t = test_pairs[idx]
                benchmark_results['Clean']['imgs'][idx] = add_scale_bar(c_t.squeeze().cpu().permute(1,2,0).numpy())
                benchmark_results['Noisy']['imgs'][idx] = add_scale_bar(n_t.squeeze().cpu().permute(1,2,0).numpy())

            # Baseline metrics
            noisy_marker = os.path.join(seed_dir, "per_image_Noisy.csv")
            if os.path.exists(noisy_marker):
                df_noisy = pd.read_csv(noisy_marker)
                per_image_store[noise_type].setdefault('Noisy', []).append(df_noisy)
                for col in ['psnr','ssim','fsim','vif','ms_ssim','fom']:
                    if col in df_noisy.columns: benchmark_results['Noisy'][col] = df_noisy[col].tolist()
            else:
                for i, (clean, noisy) in enumerate(test_pairs):
                    c_np, n_np = clean.squeeze().cpu().permute(1,2,0).numpy(), noisy.squeeze().cpu().permute(1,2,0).numpy()
                    p, s, f, v, m, fo = compute_metrics(c_np, n_np)
                    register_result('Noisy', i, c_np, n_np, p, s, f, v, m, fo, benchmark_results)
                rows = []
                for idx in range(len(benchmark_results['Noisy']['psnr'])):
                    rows.append({'seed': seed, 'noise_type': noise_type, 'method': 'Noisy', 'image_idx': idx,
                                 'psnr': benchmark_results['Noisy']['psnr'][idx], 'ssim': benchmark_results['Noisy']['ssim'][idx],
                                 'fsim': benchmark_results['Noisy']['fsim'][idx], 'vif': benchmark_results['Noisy']['vif'][idx],
                                 'ms_ssim': benchmark_results['Noisy']['msssim'][idx], 'fom': benchmark_results['Noisy']['fom'][idx]})
                df_noisy = pd.DataFrame(rows)
                df_noisy.to_csv(noisy_marker, index=False)
                per_image_store[noise_type].setdefault('Noisy', []).append(df_noisy)

            # Traditional Methods
            run_traditional_methods(test_pairs, runtimes_log[noise_type][seed], seed_dir, benchmark_results)

            def run_or_skip(method_shortname, run_fn, train_on_raw=False):
                method_key = method_shortname + ('_raw' if train_on_raw else '')
                marker_file = os.path.join(seed_dir, f"done_{method_key}.txt")
                per_image_csv = os.path.join(seed_dir, f"per_image_{method_key}.csv")
                
                if os.path.exists(marker_file):
                    if os.path.exists(per_image_csv):
                        df = pd.read_csv(per_image_csv)
                        per_image_store[noise_type].setdefault(method_key, []).append(df)
                    pn_csv = os.path.join(seed_dir, f"param_norms_{method_key}.csv")
                    if os.path.exists(pn_csv):
                        per_seed_param_norms[noise_type].setdefault(seed, {})[method_key] = pd.read_csv(pn_csv)['param_norm'].tolist()
                    l_csv = os.path.join(seed_dir, f"loss_{method_key}.csv")
                    if os.path.exists(l_csv):
                        per_seed_losses[noise_type].setdefault(seed, {})[method_key] = pd.read_csv(l_csv)['loss'].tolist()
                    r_json = os.path.join(seed_dir, "runtimes.json")
                    if os.path.exists(r_json):
                        with open(r_json, 'r') as f:
                            runtimes_log[noise_type][seed][method_key] = json.load(f).get(method_key, {})
                    return

                if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
                t_train, t_inf, params, p_norms, losses = run_fn(test_pairs, train_loader, train_on_raw=train_on_raw, benchmark_results=benchmark_results, NoiseInjector=NoiseInjector)
                
                tmp_model_path = os.path.join(Config.OUTPUT_DIR, "tmp_model_state.pth")
                if os.path.exists(tmp_model_path):
                    shutil.move(tmp_model_path, os.path.join(seed_dir, f"{method_key}_final_model.pth"))

                scale = (256.0**2) / (Config.IMG_SIZE**2)
                inf_per_256 = float(t_inf * 1000.0 * scale)
                peak_mb = float(torch.cuda.max_memory_allocated()/1e6) if torch.cuda.is_available() else 0.0
                
                runt_entry = {'train_time': float(t_train), 'inf_time_avg': float(t_inf), 'params': int(params), 'inf_time_per_256_ms': inf_per_256, 'gpu_mem_mb': peak_mb}
                runtimes_log[noise_type][seed][method_key] = runt_entry
                per_seed_param_norms[noise_type].setdefault(seed, {})[method_key] = p_norms
                per_seed_losses[noise_type].setdefault(seed, {})[method_key] = losses

                rows = []
                for idx in range(len(benchmark_results[method_key]['psnr'])):
                    rows.append({'seed': seed, 'noise_type': noise_type, 'method': method_key, 'image_idx': idx,
                                 'psnr': benchmark_results[method_key]['psnr'][idx], 'ssim': benchmark_results[method_key]['ssim'][idx],
                                 'fsim': benchmark_results[method_key]['fsim'][idx], 'vif': benchmark_results[method_key]['vif'][idx],
                                 'ms_ssim': benchmark_results[method_key]['msssim'][idx], 'fom': benchmark_results[method_key]['fom'][idx]})
                df_method = pd.DataFrame(rows)
                per_image_store[noise_type].setdefault(method_key, []).append(df_method)

                # Save artifacts
                df_method.to_csv(per_image_csv, index=False)
                pd.DataFrame({'epoch': range(len(p_norms)), 'param_norm': p_norms}).to_csv(os.path.join(seed_dir, f"param_norms_{method_key}.csv"), index=False)
                pd.DataFrame({'epoch': range(len(losses)), 'loss': losses}).to_csv(os.path.join(seed_dir, f"loss_{method_key}.csv"), index=False)
                
                r_path = os.path.join(seed_dir, "runtimes.json")
                curr_r = {}
                if os.path.exists(r_path):
                    with open(r_path, 'r') as f: curr_r = json.load(f)
                curr_r[method_key] = runt_entry
                with open(r_path, 'w') as f: json.dump(curr_r, f, indent=2)
                open(marker_file, 'w').close()

            # Self-supervised runs
            for m_fn in [run_noise2void, run_ne2ne, run_self2self, run_wrtpnet]:
                name = m_fn.__name__.replace('run_', '').upper()
                if name == 'NOISE2VOID': name = 'N2V'
                run_or_skip(name, m_fn, train_on_raw=False)
                run_or_skip(name, m_fn, train_on_raw=True)

            # Move plot images
            for img_f in glob.glob(os.path.join(Config.OUTPUT_DIR, "*.png")):
                if not os.path.basename(img_f).startswith("comp_"):
                    shutil.copy(img_f, os.path.join(seed_dir, os.path.basename(img_f)))

            # Seed zip
            s_zip = os.path.join(Config.AGGREGATED_DIR, f"results_{noise_type}_seed_{seed}.zip")
            with zipfile.ZipFile(s_zip, 'w', zipfile.ZIP_DEFLATED) as sz:
                for r, _, fs in os.walk(seed_dir):
                    for f in fs:
                        fp = os.path.join(r, f)
                        sz.write(fp, os.path.relpath(fp, seed_dir))

            for method in benchmark_results:
                if method in ['Clean', 'Noisy']: continue
                if method not in master_results[noise_type]:
                    master_results[noise_type][method] = {k: [] for k in ['psnr', 'ssim', 'fsim', 'vif', 'msssim', 'fom']}
                for metric in ['psnr', 'ssim', 'fsim', 'vif', 'msssim', 'fom']:
                    master_results[noise_type][method][metric].extend(benchmark_results[method][metric])
            last_benchmark_for_plot = (noise_type, seed, benchmark_results)

    # Aggregation
    final_rows = []
    for nt in NOISE_TYPES:
        m_image_means = {}
        for method, dfs in per_image_store[nt].items():
            if not dfs: continue
            concat = pd.concat(dfs, ignore_index=True)
            grouped = concat.groupby('image_idx')[['psnr','ssim','fsim','vif','ms_ssim','fom']].mean().reset_index()
            m_image_means[method] = grouped
            grouped.to_csv(os.path.join(Config.OUTPUT_DIR, f'per_image_mean_{nt}_{method}.csv'), index=False)
        
        for method, grouped in m_image_means.items():
            row = {'NoiseType': nt, 'Method': method}
            for met in ['psnr','ssim','fsim','vif','ms_ssim','fom']:
                v = grouped[met].values
                row[met] = f"{np.mean(v):.4f} ± {np.std(v):.4f}"
            final_rows.append(row)

        # Wilcoxon tests
        if 'Noisy' in m_image_means:
            base = m_image_means['Noisy']
            for method, grouped in m_image_means.items():
                if method == 'Noisy': continue
                wrow = {'NoiseType': nt, 'Method': method, 'ComparedTo': 'Noisy'}
                for met in ['psnr','ssim','fsim','vif','ms_ssim','fom']:
                    try:
                        a, b = base[met].values, grouped[met].values
                        mlen = min(len(a), len(b))
                        if mlen >= 5 and not np.allclose(a[:mlen], b[:mlen]):
                            s_val, p_val = stats.wilcoxon(a[:mlen], b[:mlen])
                            wrow[f"{met}_stat"], wrow[f"{met}_p"] = s_val, p_val
                    except Exception: pass
                pd.DataFrame([wrow]).to_csv(os.path.join(Config.AGGREGATED_DIR, f"wilcoxon_{nt}_{method}_vs_Noisy.csv"), index=False)

    pd.DataFrame(final_rows).to_csv(os.path.join(Config.OUTPUT_DIR, 'final_results.csv'), index=False)

    # Master Zips
    with zipfile.ZipFile(Config.MASTER_ZIP_FINAL, 'w', zipfile.ZIP_DEFLATED) as mz:
        for r, _, fs in os.walk(Config.AGGREGATED_DIR):
            for f in fs:
                if f.endswith(".zip") and f != os.path.basename(Config.ZIP_OUTPUT):
                    fp = os.path.join(r, f)
                    mz.write(fp, os.path.relpath(fp, Config.AGGREGATED_DIR))

    with zipfile.ZipFile(Config.ZIP_OUTPUT, 'w', zipfile.ZIP_DEFLATED) as zf:
        for r, _, fs in os.walk(Config.OUTPUT_DIR):
            for f in fs:
                fp = os.path.join(r, f)
                zf.write(fp, os.path.relpath(fp, Config.OUTPUT_DIR))

    # Plot comparisons
    if last_benchmark_for_plot:
        nt, sd, res = last_benchmark_for_plot
        cols = ['Clean', 'Noisy'] + sorted([m for m in res.keys() if m not in ['Clean','Noisy']])
        for idx in range(Config.NUM_PLOT_SAMPLES):
            fig, axes = plt.subplots(1, len(cols), figsize=(3*len(cols), 3))
            for i, m in enumerate(cols):
                img = res[m]['imgs'].get(idx, np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3)))
                axes[i].imshow(np.clip(img, 0, 1)); axes[i].set_title(m); axes[i].axis('off')
            plt.savefig(os.path.join(Config.OUTPUT_DIR, f'comp_{idx}.png')); plt.close()

    print("All results saved and zipped.")
