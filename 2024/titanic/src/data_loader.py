import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

def load_dataset(name: str):
    path = DATA_DIR / name
    return pd.read_csv(path)
