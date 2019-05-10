import pandas as pd, numpy as np
import fire
from pathlib import Path
from sys import stderr

def find_tokens(strs, n):
    tokens = 0
    pos = 0
    for s in strs:
        t = len(s.split())
        tokens += t
        pos += 1
        if tokens >= n:
            return pos
    print(f"Unable to find {n} tokens, only {tokens} found", file=stderr)
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
    val_end = find_tokens(z, int(1e5))
    offset = int(1e4)  # start with the same sentence
    if val_end > offset:
        print(f"Increase offset to at least {val_end}", file=stderr)
    trn_end = find_tokens(z[offset:], int(1e7))+offset
    
    to_csv(z[offset:trn_end], out_dir / "pl.wiki.train.tokens")
    to_csv(z[:val_end], out_dir / "pl.wiki.valid.tokens")
    to_csv(z[:val_end], out_dir / "pl.wiki.test.tokens")

if __name__ == "__main__": fire.Fire(split)
