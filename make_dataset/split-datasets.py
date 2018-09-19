import numpy as np
import fire
import random
from pathlib import Path
from time import time

last_time = 0

def measure(label=None):
  current = time()
  global last_time
  if last_time != 0:
    print(f"'{label}' took {current-last_time:.2f} seconds")
  last_time = current

def split_to_numpy(ids_file, output_path, valid_prec=10, test_set=False):
  measure()
  with open(ids_file, 'r') as f:
    ids = [np.array([int(x) for x in line.split()]) for line in f.readlines()]
  measure("read ids file")
  output_path = Path(output_path)
  output_path.mkdir(parents=True, exist_ok=True)
  if test_set:
    np.save(output_path / 'test_ids.npy', np.array(ids))
    return

  token_count = sum(len(s) for s in ids)
  random.seed(12345)
  random.shuffle(ids)
  measure("shuffle")
  valid = 0
  valid_idx = 0
  valid_size = int(token_count*valid_prec/100.0)
  print(f"Traning data sentences: {len} size in tokens: {token_count}, valid size: {valid_size}")
  while valid < valid_size:
    valid += len(ids[valid_idx])
    valid_idx += 1
  measure("get valid")
  train = 0

  np.save(output_path / 'val_ids.npy', np.array(ids[0:valid_idx]))
  measure("save valid")
  np.save(output_path / 'trn_ids.npy', np.array(ids[valid_idx:]))
  measure("save train")

if __name__ == '__main__': fire.Fire(split_to_numpy)
