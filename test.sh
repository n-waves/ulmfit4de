#!/usr/bin/env bash

TESTFILE="$1"
[ ! -e "$TESTFILE" ] && echo "'$TESTFILE' does not exist" 1>&2 && exit 1

python preprocessing/reddit-preprocess.py --filename "$TESTFILE" --format tsv --keep-ogonki True --lowercase False --keep-numbers False --escape-emoji True --remove-asciiart False --output-format csv --twitter | python tsv2ids.py --input-path work/hatepl --output-path work/hatepl --test --labels False

python ./ulmfit/evaluate.py --dir-path="work/hatepl" --cuda-id=0 --clas-id="nl-4-v1_best" --bs=120 --nl 4  --test-file test --classes 2 --dump-preds poleval19-t6-s1.txt

cat poleval19-t6-s1.txt
