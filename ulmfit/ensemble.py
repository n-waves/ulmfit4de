from functools import reduce

import fire
from fastai.text import *
from fastai.lm_rnn import *
import sentencepiece as sp
from scipy import stats
import fastai.core
from sklearn.metrics import confusion_matrix
from fastai.plots import plot_confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

# In[3]:


def print_metrics(true_y, preds, f1_limit=False):
    cm = confusion_matrix(true_y, preds)
    f1_micro_avg = f1_score(true_y, preds, average='micro')
    f1_macro_avg = f1_score(true_y, preds, average='macro')
    pa = precision_score(true_y, preds, average='macro')
    ra = recall_score(true_y, preds, average='macro')
    if not f1_limit:
        print("F1 score micro avg:", f1_micro_avg)
        print("F1 score macro avg:", f1_macro_avg)
        print("Precision macro avg:", pa)
        print("Recall macro avg:", ra)
        print("Confusion matrix\n", cm)
    print("Special F1 macro* avg:", 2/(1/pa + 1/ra))

    if np.max(true_y) == 1 and not f1_limit:
        f1_bin = f1_score(true_y, preds, average='binary')
        prec = precision_score(true_y, preds)
        recall = recall_score(true_y, preds)
        print("Binary")
        print("F1 score bin:", f1_bin)
        print("Precision:", prec)
        print("Recall:", recall)

def read_prop(fn, key="f1_macro_avg_"):
    with open(fn.with_suffix(".json"), "r") as f:
        props = json.load(f)
    return props[key]

def ensemble(dir_path, test_file='test'):
    p = Path(dir_path)
    ensemble = p/"models"/test_file

    lbl = np.load(p / "tmp" / f"lbl_{test_file}.npy")

    files = list(ensemble.glob("*.npy"))



    en_preds = reduce(np.add, [np.load(f)  for f in files])
    en_preds = np.argmax(en_preds, axis=1)

    print("Final")
    print_metrics(lbl, en_preds)



if __name__ == '__main__': fire.Fire(ensemble)