# Pipline to train German language model and sentiment classifier

This is early commits based on the Poleval2018, it won't work well for the time being.

Our solution is an extension of the work done by FastAI team to train language models for English.
We extended it with google sentence piece to tokenize German words. 


## Installation
The source code needs cleaning up to minimise the amount of work needed to run it.

But for now here are rough manual steps:

- Install fastai from [our fork](https://github.com/n-waves/fastai/releases/tag/poleval2018) (python PATH) 
- Install sentencepiece from [source code](https://github.com/google/sentencepiece/) (PATH and python PATH)

Requirements
- jq  - `apt install jq`

## Training
You should have the following structure:
```
.
├── data
│   ├── germeval2017
│   │   ├── dev_v1.4.tsv
│   │   ....
│   │   ├── train_v1.4.tsv
│   │   └── train_v1.4.xml
│   ├── recorded-tweets.zip
│   └── btw17
│       ├── ...
├── make_dataset
└── README.md
└── work  # this will be created by scripts
    ├── nouniq
    │   ├── models
    │   └── tmp
    └── up_low50k
        ├── models
        └── tmp 
```

# Workflow

To create data set:
```bash
cd make_dataset
WORK_DIR="../work"
CACHE_DIR="${WORK_DIR}/shared"
DICT_SIZE=30
./prepare-data.sh --work-dir "${WORK_DIR}/btw-nouniq${DICT_SIZE}k" --cache-dir "${CACHE_DIR}" --vocab-size "${DICT_SIZE}000" --model-name "sp" --most-low "False" --lower-case "False" --uniq "False"
```

To start training lm model
```bash
dir=work/btw-nouniq30k
BS=192
nl=4
cuda=0
python fastai_scripts/pretrain_lm.py --dir-path "${dir}" --cuda-id $cuda --cl 12 --bs "${BS}" --lr 0.01 --pretrain-id "nl-${nl}-small-minilr" --sentence-piece-model sp.model --nl "${nl}"
```

To see the perplexity of the model on a test set.
```
python fastai_scripts/infer.py --dir-path "${dir}" --cuda-id $cuda --bs 22 --pretrain-id "nl-${nl}-small-minilr" --sentence-piece-model sp.model --test_set tmp/val_ids.npy --correct_for_up=False --nl  "${nl}"
```

To fine tune
```
BS=128
python ./fastai_scripts/finetune_lm.py --dir-path "${dir}" --pretrain-path "${dir}" --cuda-id $cuda \
    --cl 6 --pretrain-id "nl-${nl}-small-minilr" --lm-id "nl-${nl}-finetune" --bs $BS --lr 0.001 \
    --use_discriminative False --dropmult 0.5 --sentence-piece-model sp.model --sampled True --nl "${nl}"
```

```bash
BS=192
nl=4
cuda=0
python ./fastai_scripts/finetune_lm.py --dir-path "work/ge2017" --pretrain-path "work/btw-nouniq30k" --cuda-id $cuda \
    --cl 6 --pretrain-id "nl-${nl}-small-minilr" --lm-id "nl-${nl}-ge2017" --bs $BS --lr 0.001 \
    --use_discriminative False --dropmult 0.5 --sentence-piece-model sp.model --sampled True --nl "${nl}"

```
```bash

# discriminative
python ./fastai_scripts/train_clas.py --dir-path="work/ge2017" --cuda-id=$cuda \
    --lm-id="nl-${nl}-ge2017-all" --clas-id="class-nl-${nl}-ge2017"\
    --bs=$BS --cl=5 --lr=0.01 --dropmult 0.5 --sentence-piece-model='sp.model' --nl 4 --use_discriminative False
```

```bash
BS=128
nl=4
cuda=1
python ./fastai_scripts/train_clas.py --dir-path="work/ge2017" --cuda-id=$cuda \
    --lm-id="nl-${nl}-ge2017-all" --clas-id="class-nl-${nl}-ge2017"\
    --bs=$BS --cl=5 --lr=0.01 --dropmult 0.5 --sentence-piece-model='sp.model' --nl 4 --use_discriminative True
    
```
python ./fastai_scripts/train_clas.py --dir-path="work/ge2017" --cuda-id=2 \
    --lm-id="nl-4-ge2017-all" --clas-id="class2-nl-4-ge2017"\
    --bs=40 --cl=5 --lr=0.001 --dropmult 0.5 --sentence-piece-model='sp.model' \
    --nl 4 --use_discriminative True
    
    
    
```bash
destdir=work/ge2017
BS=120
cuda=0
nl=4
python ./ulmfit/evaluate.py --dir-path="$destdir" --cuda-id=$cuda \
    --clas-id="class2-nl-${nl}-ge2017" --bs=$BS --nl "${nl}"
```
