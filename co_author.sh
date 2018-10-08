#!/bin/sh

DISTFN=$1
DIM=$2
SEED=$3
NAME="co_author_0.01_${DISTFN}_${DIM}_${SEED}"
EXP=co_author
DIR=DBLP
NPROC=1
NDPROC=2
NEPROC=30
NEGS=10
BATCH=64
HIDDENDIM=10000
HIDDENLNUM=1
LR=0.01
ITER=300000
EACH=5000

export OMP_NUM_THREADS=1

time python main.py \
-name "${NAME}" \
-seed "${SEED}" \
-exp "${EXP}" \
-dsetdir "${DIR}" \
-dim "${DIM}" \
-hidden_size "${HIDDENDIM}" \
-hidden_layer_num "${HIDDENLNUM}" \
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
