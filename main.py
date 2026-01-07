import os
import torch
import random
import numpy as np
from dataset import BenchmarkDataset, NoiseInjector
from engine import run_training_method
from torch.utils.data import DataLoader

class Config:
    INPUT_DIR = "./data"
    OUTPUT_DIR = "./results"
    IMG_SIZE = 256
    BATCH_SIZE = 8
    EPOCHS = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEEDS = [42, 100, 123]

def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    # Main logic loop over noise types and seeds...
    pass

if __name__ == "__main__":
    main()
