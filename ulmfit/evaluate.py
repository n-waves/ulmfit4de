import fire
from fastai.text import *
from fastai.lm_rnn import *
import sentencepiece as sp
from scipy import stats
import fastai.core
from sklearn.metrics import confusion_matrix
from fastai.plots import plot_confusion_matrix
from sklearn.metrics import f1_score

# In[3]:


def evaluate_model(test_file, m, p, bs=120):
    tst = np.load(p / f"{test_file}_ids.npy")
    lbl = np.load(p / f"lbl_{test_file}.npy")

    m.reset()
    m.eval()

    tst_ds = TextDataset(tst, lbl)
    tst_samp = SortSampler(tst, key=lambda x: len(tst[x]))
    tst_dl = DataLoader(tst_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=tst_samp)

    res = predict_with_targs(m, tst_dl)

    preds = np.argmax(res[0], axis=1)
    true_y = res[1]

    cm = confusion_matrix(true_y, preds)
    f1 = f1_score(true_y, preds, average='micro')
    print("Test file:", test_file)
    print("F1 score:", f1)
    print("Confusion matrix\n", cm)
    return f1, cm

def evaluate(dir_path, clas_id, test_file='test1', cuda_id=0, nl=4, classes=3, bs=120):
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

    return evaluate_model(test_file, m, p/"tmp", bs)

if __name__ == '__main__': fire.Fire(evaluate)