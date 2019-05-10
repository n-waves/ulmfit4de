import pandas as pd, numpy as np
import fire
from pathlib import Path

def find_tokens(strs, n):
    tokens = 0
    pos = 0
    for s in strs:
        t = len(s.split())
        tokens += t
        pos += 1
        if tokens >= n:
            return pos
    return pos

def to_csv(strs, path):
    pd.DataFrame(strs).to_csv(path, header=None, index=None)

def split(data_dir):
    data_dir = Path(data_dir)
    out_dir = data_dir / "pl-10"
    out_dir.mkdir(exist_ok=True)
    np.random.seed(12345)
    x = pd.read_csv(data_dir / "cleaned.csv", header=None).fillna('')
    y = x[0].values
    z = np.random.choice(y, len(y), replace=False)
    start = find_tokens(z, int(1e7))
    end = find_tokens(z[start:], int(1e5))+start
    
    to_csv(z[:start], out_dir / "pl.wiki.train.tokens")
    to_csv(z[start:end], out_dir / "pl.wiki.valid.tokens")
    to_csv(z[start:end], out_dir / "pl.wiki.test.tokens")

if __name__ == "__main__": fire.Fire(split)
