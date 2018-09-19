# Pipline to train German language model and sentiment classifier

This is early commits based on the Poleval2018, it won't work well for the time being.

Our solution is an extension of the work done by FastAI team to train language models for English.
We extended it with google sentence piece to tokenize German words. 


## Installation
The source code needs cleaning up to minimise the amount of work needed to run it.

But for now here are rough manual steps:

- Install fastai from [our fork](https://github.com/n-waves/fastai/releases/tag/poleval2018) (python PATH) 
- Install sentencepiece from [source code](https://github.com/google/sentencepiece/) (PATH and python PATH)

## Training
You should have the following structure:
```
.
├── data
│   └── task3 # here goes unzipped files
│       ├── test
│       └── train
├── fastai_scripts -> github.com/fastai/fastai/courses/dl2/imdb_scripts/ (our version)
├── task3
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
./prepare-data.sh --work-dir "${WORK_DIR}/nouniq${DICT_SIZE}k" --cache-dir "${CACHE_DIR}" --vocab-size "${DICT_SIZE}000" --model-name "sp" --most-low "False" --lower-case "False" --uniq "False"
```

To start training lm model
```bash
dir=work/nouniq25k
BS=192
nl=4
cuda=0
python fastai_scripts/pretrain_lm.py --dir-path "${dir}" --cuda-id $cuda --cl 12 --bs "${BS}" --lr 0.01 --pretrain-id "nl-${nl}-small-minilr" --sentence-piece-model sp.model --nl "${nl}"
```

To see the perplexity of the model on a test set.
```
python fastai_scripts/infer.py --dir-path "${dir}" --cuda-id $cuda --bs 22 --pretrain-id "nl-${nl}-small-minilr" --sentence-piece-model sp.model --test_set tmp/test_ids.npy --correct_for_up=False --nl  "${nl}"
```

To fine tune
```
BS=128
python ./fastai_scripts/finetune_lm.py --dir-path "${dir}" --pretrain-path "${dir}" --cuda-id $cuda \
    --cl 6 --pretrain-id "nl-${nl}-small-minilr" --lm-id "nl-${nl}-finetune" --bs $BS --lr 0.001 \
    --use_discriminative False --dropmult 0.5 --sentence-piece-model sp.model --sampled True --nl "${nl}"
```