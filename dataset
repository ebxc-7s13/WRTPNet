import glob
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class BenchmarkDataset(Dataset):
    def __init__(self, dir_path, img_size=256):
        self.files = sorted(glob.glob(os.path.join(dir_path, "**", "*.png"), recursive=True))
        self.img_size = img_size
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        return torch.from_numpy(img.transpose(2,0,1)).float() / 255.0

class NoiseInjector:
    @staticmethod
    def add_gaussian(img, sigma=0.1):
        return (img + torch.randn_like(img) * sigma).clamp(0,1)
    @staticmethod
    def add_poisson(img, peak=30.0):
        return (torch.poisson(img * peak) / peak).clamp(0,1)
