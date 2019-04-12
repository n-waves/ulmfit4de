#!/usr/bin/env bash
set -e -x

[ -z "${CUDA_VERSION+x}" ] && echo "Set CUDA_VERSION, f.e., CUDA_VERSION=9.0 ./setup-ulmfit4pl.sh" 1>&2 && exit 1

[ ! -d "ulmfit4pl" ] && git clone -b poleval19/hatespeech --single-branch "https://github.com/n-waves/ulmfit4de" "ulmfit4pl"
[ ! -d "fastai" ] && git clone -b poleval19/hatespeech --single-branch "https://github.com/n-waves/fastai"

conda create -y -n ulmfit4pl -c pytorch python=3.7 pytorch=0.4.1 cudatoolkit="${CUDA_VERSION}" scikit-learn
source activate ulmfit4pl

cd fastai && pip install -e '.' && cd ..
pip install sentencepiece fire sacremoses Unidecode

