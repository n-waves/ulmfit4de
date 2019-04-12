import sys
import csv
import numpy as np
import sentencepiece as sp
from pathlib import Path
from sklearn.model_selection import train_test_split
import fire


def tokenize(input_path, output_path, test=False, labels=True):
    tabin = csv.reader(sys.stdin, dialect=csv.excel_tab)
    rows = list(tabin)
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    spp = sp.SentencePieceProcessor()
    spp.Load(str( input_path / 'tmp' / 'sp.model'))
    spp.SetEncodeExtraOptions("bos:eos")
    
    ids = np.array([np.array(spp.EncodeAsIds(x[0])) for x in rows])
    if labels:
        lbl = np.array([[int(x[1]!='OTHER')] for x in rows])
   
    if not test:
        if labels:
            trn_ids, val_ids, trn_lbl, val_lbl = train_test_split(ids, lbl, test_size=0.1, random_state=12345)
        else:
            trn_ids, val_ids = train_test_split(ids, test_size=0.1)
        np.save(output_path / 'tmp' / 'val_ids.npy', val_ids)
        np.save(output_path / 'tmp' / 'trn_ids.npy', trn_ids)
        if labels:
            np.save(output_path / 'tmp' / 'lbl_val.npy', val_lbl)
            np.save(output_path / 'tmp' / 'lbl_trn.npy', trn_lbl)
    else:
        np.save(output_path / 'tmp' / 'test_ids.npy', ids)
        if labels:
            np.save(output_path / 'tmp' / 'lbl_test.npy', lbl)
        
    
if __name__ == '__main__': fire.Fire(tokenize)
