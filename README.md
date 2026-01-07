# WRTPNet
A Self-Supervised Blind-Spot Wavelet Residual Network for Label-Free Autofluorescent Image Denoising

Fairness Benchmarking for Self-Supervised Denoising

This repository provides a standardized framework for evaluating self-supervised denoising models. By using identical training budgets, datasets, and noise injections, this suite ensures a "fair" comparison between popular architectures and traditional algorithms.

Features

Unified Pipeline: Compare N2V, Neighbor2Neighbor, Self2Self, and WRTPNet in a single run.

Diverse Noise Profiles: Supports Gaussian, Poisson, and Poisson-Gaussian (mixed) noise.

Scientific Metrics: Automated calculation of PSNR, SSIM, MS-SSIM, FSIM, VIF, and FOM.

Statistical Validation: Generates Wilcoxon signed-rank tests to verify if improvements are statistically significant.

Reproducibility: Saves model weights (.pth) and experiment logs for every seed and noise type.

Installation

First, clone the repository and install the necessary dependencies:

git clone [[https://github.com/Jacdencity/WRTPNet.git](https://github.com/Jacdencity/WRTPNet).
cd denoising-fairness
pip install -r requirements.txt


Dataset Structure

Place your clean images in a folder. The script will recursively find all images (PNG, JPG, BMP, TIF).

/data
  ├── train/
  │    ├── image1.png
  │    └── image2.png
  └── test/
       ├── sample1.png
       └── sample2.png


Usage

To run the full benchmark across all noise types and seeds defined in Config:

'''python main.py'''


Developed for the research community to promote transparent and reproducible denoising benchmarks.
