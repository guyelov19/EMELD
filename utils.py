import os
import tarfile
import requests
import pandas as pd
import torch
from config import TRAIN_PATH, TEST_PATH, HF_TOKEN
from huggingface_hub import login

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_dataset():
    """Loads dataset CSV files into pandas DataFrames."""
    return pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)

# Hugging Face authentication
login(token=HF_TOKEN)
