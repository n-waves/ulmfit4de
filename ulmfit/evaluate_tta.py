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
OTHER=0
OFFENSE=1
def error_analysis(preds, true_y, tst_dl, spp, backwards):
    tweets = np.array([[int(i) for i in tweet] for *x, y in iter(tst_dl) for tweet in x[0].cpu().numpy().transpose()])
    d = -1 if backwards else 1
    true_y = true_y[:, 0]

    offenses = true_y == OFFENSE
    for s, claz in [('offenses', OFFENSE), ('neutrals', OTHER)]:
        print(f'True {s}:')
        probs = torch.from_numpy(preds).softmax(1).numpy()[:, claz]
        true_claz = true_y == claz
        pos = probs[true_claz].argsort()[::-1]
        for i in pos[:5]:
            print(f"{probs[true_claz][i]:0.2f}: {spp.DecodeIds(tweets[true_claz][i][::d])}")
        print("...")
        m = len(pos) // 2
        for i in pos[m-2:m+3]:
            print(f"{probs[true_claz][i]:0.2f}: {spp.DecodeIds(tweets[true_claz][i][::d])}")
        print("...")
        for i in pos[-5:]:
            print(f"{probs[true_claz][i]:0.2f}: {spp.DecodeIds(tweets[true_claz][i][::d])}")

class SPDataset(Dataset):
  def __init__(self, spp, sentences, labels):
    self.spp = spp
    self.sentences = sentences
    self.labels = labels

  def __len__(self):
    return len(self.sentences)

class SampleEncodeDataset(SPDataset):
  def __init__(self, spp, sentences, labels, alpha=0.1, n=64):
    super().__init__(spp, sentences, labels)
    self.alpha = alpha
    self.n = n
  def __getitem__(self, index):
    return np.array(self.spp.SampleEncodeAsIds(self.sentences[index], self.n, self.alpha)), self.labels[index]

class BestEncodeDataset(SPDataset):
  def __getitem__(self, index):
    return np.array(self.spp.EncodeAsIds(self.sentences[index])), self.labels[index]

def evaluate_model(test_file, m, p, spp, sp_alpha=0.1, sp_n=64, n_tta=7, best=True, bs=120, squeeze_bin=False, backwards=False):
    with open(p / f'test.txt', 'r') as f:
        rows = [line.split('\t') for line in f.readlines()]
    tst = [row[0] for row in rows]
    lbl = np.array([[int(row[1] != 'OTHER')] for row in rows])

    m.reset()
    m.eval()

    if best:
        tst_ds = BestEncodeDataset(spp, tst, lbl)
        n_tta = 1
    else:
        tst_ds = SampleEncodeDataset(spp, tst, lbl, sp_alpha, sp_n)
    tst_samp = SortSampler(tst, key=lambda x: len(tst[x]))
    tst_dl = DataLoader(tst_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=tst_samp)

    pr = np.zeros(len(tst))
    for i in range(n_tta):
        res = predict_with_targs(m, tst_dl)
        pr += torch.from_numpy(res[0]).softmax(1).numpy()[:, 1]
    true_y = res[1]
    order = np.array(list(tst_samp))
    #preds = np.argmax(res[0], axis=1)
    preds = (pr / n_tta > 0.5).astype(int)

    if squeeze_bin:
        print ("Converting mutli classification in to binary classificaiton")
        preds = preds > 0
        true_y = true_y > 0

    error_analysis(res[0], true_y, tst_dl, spp, backwards)
    cm = confusion_matrix(true_y, preds)
    f1_micro_avg = f1_score(true_y, preds, average='micro')
    f1_macro_avg = f1_score(true_y, preds, average='macro')
    pa = precision_score(true_y, preds, average='macro')
    ra = recall_score(true_y, preds, average='macro')
    print("Test file:", test_file)
    print("Sum of all 1s lbls", np.sum(true_y))
    print("Sum of all 1s preds", np.sum(preds))
    print("F1 score micro avg:", f1_micro_avg)
    print("F1 score macro avg:", f1_macro_avg)
    print("Precision macro avg:", pa)
    print("Recall macro avg:", ra)
    print("Special F1 macro* avg:", 2/(1/pa + 1/ra))
    
    for i in [0,1]:
      p = precision_score(true_y, preds, pos_label=i)
      r = recall_score(true_y, preds, pos_label=i)
      print(f"Precision[{i}]={p}")
      print(f"Recall[{i}]={r}")

    if np.max(true_y) == 1:
        f1_bin = f1_score(true_y, preds, average='binary')
        prec = precision_score(true_y, preds)
        recall = recall_score(true_y, preds)
        print("Binary")
        print("F1 score bin:", f1_bin)
        print("Precision:", prec)
        print("Recall:", recall)
    print("Confusion matrix\n", cm)
    return (res[0])[np.argsort(order)], {
        "f1_micro_avg_":f1_micro_avg,
        "f1_macro_avg_":f1_macro_avg,
        "pred_macro": pa,
        "recall_macro": ra,
    }

def evaluate(dir_path, clas_id, test_file='test1', cuda_id=0, nl=4, classes=3, bs=120, sp_alpha=0.1, sp_n=64, tta=7, best=True, squeeze_bin=False, backwards=False):
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)

    p = Path(dir_path)
    spp = sp.SentencePieceProcessor()
    spp.Load(str(p /"tmp" / 'sp.model'))
    vs = spp.GetPieceSize()  # len(itos)
    spp.SetEncodeExtraOptions("bos:eos:reverse" if backwards else "bos:eos")

    # In[14]:
    dps = np.array([0.4,0.5,0.05,0.3,0.4])
    bptt,em_sz,nh,nl = 70,400,1150,nl
    c=classes
    m = get_rnn_classifier(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
              layers=[em_sz*3, 50, c], drops=[dps[4], 0.1],
              dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])
    PRE = 'bwd_' if backwards else 'fwd_'
    model_path = p/"models"/f"{PRE}{clas_id}_clas_1.h5"
    load_model(m, model_path)
    print("Loading", model_path)
    m = to_gpu(m)
    direction="bwd" if backwards else "fwd"
    preds, metrics = evaluate_model(test_file, m, p/"tmp", spp, sp_alpha, sp_n, best, tta, bs, squeeze_bin, backwards)
    (p / "models" / test_file ).mkdir(exist_ok=True)
    np.save(p/"models"/ test_file/ f"{direction}_{clas_id}_clas_1-results.npy", preds)

    with open(p / "models" / test_file / f"{direction}_{clas_id}_clas_1-results.json", 'w') as fp:
        json.dump(metrics, fp)
if __name__ == '__main__': fire.Fire(evaluate)
