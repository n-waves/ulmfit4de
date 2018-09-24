# wikipedia
#### Fetch data
I've used the script prepare_wiki_now.sh,
Then I've used python shell to convert csv to text and added btw17 to the wikipedia text
```py
train_wiki = pd.read_csv(wiki/"train.csv", header=None)
np.savetxt(wiki/'train.txt', train_wiki.values, fmt="%s")
!cat {BTW17}/text.txt >> {wiki}/train.txt
```
 
#### prepare dataset
```bash
cd make_dataset
WORK_DIR="../work"
CACHE_DIR="${WORK_DIR}/shared-wiki"
DICT_SIZE=30
./prepare-data-wiki.sh --work-dir "${WORK_DIR}/wiki${DICT_SIZE}k" --cache-dir "${CACHE_DIR}" --vocab-size "${DICT_SIZE}000" --model-name "sp" --most-low "False" --lower-case "False" --uniq "False"
```

#### pretrain
```bash
dir=work/wiki30k
BS=192
nl=4
cuda=0
python fastai_scripts/pretrain_lm.py --dir-path "${dir}" --cuda-id $cuda --cl 12 --bs "${BS}" --lr 0.01 --pretrain-id "nl-${nl}-small-minilr" --sentence-piece-model sp.model --nl "${nl}"
```

```bash
dir=work/wiki30k
BS=20
nl=4
cuda=0
python fastai_scripts/infer.py --dir-path "${dir}" --cuda-id $cuda --bs $BS\
    --pretrain-id "nl-${nl}-small-minilr" --sentence-piece-model sp.model \
    --test_set tmp/val_ids.npy --correct_for_up=False --nl  "${nl}"
```

#### fine tune
```bash
predir=work/wiki30k
destdir=work/wikige2017
BS=128
cuda=0
nl=4
python ./fastai_scripts/finetune_lm.py --dir-path "${destdir}" --pretrain-path "${predir}" --cuda-id $cuda \
    --cl 6 --pretrain-id "nl-${nl}-small-minilr" --lm-id "nl-${nl}-finetune" --bs $BS --lr 0.001 \
    --use_discriminative True --dropmult 0.5 --sentence-piece-model sp.model --sampled True --nl "${nl}"
    
```

#### train classifier
```bash
predir=work/wiki30k
destdir=work/wikige2017
BS=40
cuda=0
nl=4
python ./fastai_scripts/train_clas.py --dir-path="$destdir" --cuda-id=$cuda \
    --lm-id="nl-${nl}-finetune" --clas-id="nl-${nl}-v1"\
    --bs=$BS --cl=5 --lr=0.001 --dropmult 0.5 --sentence-piece-model='sp.model' \
    --nl $nl --use_discriminative True
```

```bash
destdir=work/wikige2017
BS=120
cuda=0
nl=4
python ./ulmfit/evaluate.py --dir-path="$destdir" --cuda-id=$cuda \
    --clas-id="nl-${nl}-v1" --bs=$BS --nl $nl
    
python ./ulmfit/evaluate.py --dir-path="$destdir" --cuda-id=$cuda \
    --clas-id="nl-${nl}-v1" --bs=$BS --nl $nl --test-file test2
```


# wikipedia backward
    
#### prepare dataset
```it uses notebook: backward```

#### pretrain
```bash
dir=work/wiki30k
BS=192
nl=4
cuda=0
python fastai_scripts/pretrain_lm.py --dir-path "${dir}" --cuda-id $cuda\
 --cl 12 --bs "${BS}" --lr 0.01 --pretrain-id "nl-${nl}-base"\
  --sentence-piece-model sp.model --nl "${nl}" --backwards True
```

```bash
dir=work/wiki30k
BS=20
nl=4
cuda=0
python fastai_scripts/infer.py --dir-path "${dir}" --cuda-id $cuda --bs $BS\
    --pretrain-id "nl-${nl}-small-minilr" --sentence-piece-model sp.model \
    --test_set tmp/val_ids.npy --correct_for_up=False --nl  "${nl}"
```

#### fine tune
```bash
predir=work/wiki30k
destdir=work/wikige2017
BS=128
cuda=0
nl=4
python ./fastai_scripts/finetune_lm.py --dir-path "${destdir}" --pretrain-path "${predir}" --cuda-id $cuda \
    --cl 6 --pretrain-id "nl-${nl}-small-minilr" --lm-id "nl-${nl}-finetune" --bs $BS --lr 0.001 \
    --use_discriminative True --dropmult 0.5 --sentence-piece-model sp.model --sampled True --nl "${nl}"
    
```

#### train classifier
```bash
predir=work/wiki30k
destdir=work/wikige2017
BS=40
cuda=0
nl=4
python ./fastai_scripts/train_clas.py --dir-path="$destdir" --cuda-id=$cuda \
    --lm-id="nl-${nl}-finetune" --clas-id="nl-${nl}-v1"\
    --bs=$BS --cl=5 --lr=0.001 --dropmult 0.5 --sentence-piece-model='sp.model' \
    --nl $nl --use_discriminative True
```

```bash
destdir=work/wikige2017
BS=120
cuda=0
nl=4
python ./ulmfit/evaluate.py --dir-path="$destdir" --cuda-id=$cuda \
    --clas-id="nl-${nl}-v1" --bs=$BS --nl $nl
```
