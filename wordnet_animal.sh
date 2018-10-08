#!/bin/sh

DISTFN=$1
DIM=$2
SEED=$3
EXP="wordnet"
DIR="wordnet"
NPROC=1
NDPROC=4
NEPROC=10
NEGS=20
BATCH=128
LR=0.001
ITER=150000
EACH=5000
NAME="wordnet_animal_0.001_${DISTFN}_${DIM}_${SEED}"

export OMP_NUM_THREADS=1

time python main.py \
-name "${NAME}" \
-seed "${SEED}" \
-exp "${EXP}" \
-dsetdir "${DIR}" \
-dim "${DIM}" \
-lr "${LR}" \
-iter "${ITER}" \
-eval_each "${EACH}" \
-negs "${NEGS}" \
-nproc "${NPROC}" \
-ndproc "${NDPROC}" \
-neproc "${NEPROC}" \
-distfn "${DISTFN}" \
-batchsize "${BATCH}" \
-undirect
