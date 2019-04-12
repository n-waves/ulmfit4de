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


def evaluate_model(test_file, m, p, spp, bs=120, squeeze_bin=False, backwards=False, labels=False):
    tst = np.load(p / f"{test_file}_ids.npy")
    if labels:
        lbl = np.load(p / f"lbl_{test_file}.npy")
    else:
        lbl = np.zeros(len(tst), dtype=np.int)

    m.reset()
    m.eval()

    tst_ds = TextDataset(tst, lbl)
    tst_samp = SortSampler(tst, key=lambda x: len(tst[x]))
    tst_dl = DataLoader(tst_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=tst_samp)

    res = predict_with_targs(m, tst_dl)
    order = np.array(list(tst_samp))
    true_y = res[1]
    preds = np.argmax(res[0], axis=1)

    if squeeze_bin:
        print ("Converting mutli classification in to binary classificaiton")
        preds = preds > 0
        true_y = true_y > 0

    if labels:
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
    else:
        return (res[0])[np.argsort(order)], {}

def evaluate(dir_path, clas_id, test_file='test1', cuda_id=0, nl=4, classes=3, bs=120, squeeze_bin=False, backwards=False, dump_preds=None, labels=False):
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)

    p = Path(dir_path)
    spp = sp.SentencePieceProcessor()
    spp.Load(str(p /"tmp" / 'sp.model'))
    vs = spp.GetPieceSize()  # len(itos)
    spp.SetEncodeExtraOptions("bos:eos")


    # In[14]:
    dps = np.array([0.4,0.5,0.05,0.3,0.4])
    bptt,em_sz,nh,nl = 70,400,1150,nl
    c=classes
    m = get_rnn_classifier(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
              layers=[em_sz*3, 50, c], drops=[dps[4], 0.1],
              dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])
    model_path = p/"models"/f"fwd_{clas_id}_clas_1.h5"
    load_model(m, model_path)
    print("Loading", model_path)
    m = to_gpu(m)
    direction="bwd" if backwards else "fwd"
    preds, metrics = evaluate_model(test_file, m, p/"tmp", spp, bs, squeeze_bin, backwards, labels)
    if dump_preds is not None:
        with open(dump_preds, 'w') as f:
            f.write('\n'.join([str(int(x>=0)) for x in preds[:, 1]]))
    (p / "models" / test_file ).mkdir(exist_ok=True)
    np.save(p/"models"/ test_file/ f"{direction}_{clas_id}_clas_1-results.npy", preds)

    with open(p / "models" / test_file / f"{direction}_{clas_id}_clas_1-results.json", 'w') as fp:
        json.dump(metrics, fp)
if __name__ == '__main__': fire.Fire(evaluate)
