import pandas as pd
from pathlib import Path

def load_data(train_path="data/train.csv", test_path="data/test.csv", sample_path="data/sample_submission.csv"):
    base_dir = Path(__file__).resolve().parents[1]
    print("Looking in:", base_dir / train_path)   # <-- debug line
    train = pd.read_csv(base_dir / train_path)
    test = pd.read_csv(base_dir / test_path)
    sample = pd.read_csv(base_dir / sample_path)
    return train, test, sample
