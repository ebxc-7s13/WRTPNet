# WRTPNet
A Self-Supervised Blind-Spot Wavelet Residual Network for Label-Free Autofluorescent Image Denoising

Fairness Benchmarking for Self-Supervised Denoising

This repository contains the implementation and benchmarking suite for evaluating self-supervised denoising models (N2V, Neighbor2Neighbor, Self2Self, and WRTPNet) across various noise distributions (Gaussian, Poisson, Poisson-Gaussian).

Features

Modular Design: Separated models, datasets, and training engines.

Fair Comparison: Identical training budgets and evaluation metrics for all methods.

Comprehensive Metrics: Includes PSNR, SSIM, MS-SSIM, FSIM, VIF, and FOM.

Statistical Analysis: Automatic Wilcoxon signed-rank tests for result significance.

Installation

git clone [https://github.com/yourusername/denoising-fairness.git](https://github.com/yourusername/denoising-fairness.git)
cd denoising-fairness
pip install -r requirements.txt


Dataset Structure

The script expects a directory of images. Default path is /data.

/data
  ├── image1.png
  ├── image2.png
  └── ...


Usage

Run the full benchmark suite with:

python main.py
