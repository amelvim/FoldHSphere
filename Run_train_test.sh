#!/bin/bash


###############################################################################
# Define arguments
PHASE="train"
# train / test
NET="CNN-GRU"
# CNN-GRU / CNN-BGRU / ResCNN-GRU / ResCNN-BGRU
ACTIV="tanh"
# sigmoid / tanh
LOSSTYPE="lmcl"
# softmax / lmcl / lmcl_fixed
MARGIN=25
SCALE=30
EPTHOMSON=1130
NWORKERS=1
###############################################################################

# Set arguments for model type
if [[ $NET == "CNN-GRU" ]]; then
    MODELTYPE="cnn_gru"; CHDIMS="128_256"; BIDIREC=False;
elif [[ $NET == "CNN-BGRU" ]]; then
    MODELTYPE="cnn_gru"; CHDIMS="128_256"; BIDIREC=True;
elif [[ $NET == "ResCNN-GRU" ]]; then
    MODELTYPE="rescnn_gru"; CHDIMS="64_256_64_256"; BIDIREC=False;
elif [[ $NET == "ResCNN-BGRU" ]]; then
    MODELTYPE="rescnn_gru"; CHDIMS="64_256_64_256"; BIDIREC=True;
fi

# Set arguments for model loss type and define model directory
if [[ $LOSSTYPE == "softmax" ]]; then   CENTFILE="";
    MODELDIR="models/${NET}/${ACTIV}_softmax";
elif [[ $LOSSTYPE == "lmcl" ]]; then    CENTFILE="";
    MODELDIR="models/${NET}/${ACTIV}_lmcl_m0${MARGIN}_s${SCALE}";
elif [[ $LOSSTYPE == "lmcl_fixed" ]]; then
    MODELDIR="models/${NET}/${ACTIV}_thomson_lmcl_m0${MARGIN}_s${SCALE}"
    CENTDIR="models/prototypes_thomson/optim_${NET,,}-softmax/thl_sum"
    CENTFILE="${CENTDIR}/prototypes_ep${EPTHOMSON}.npy"
fi

mkdir -p ${MODELDIR}


###############################################################################
# Training phase
if [[ $PHASE == "train" ]]; then

TRAINFILE="data/train/train.list"
FOLDLABELFILE="data/train/fold_label_relation_1154.txt"
FEATSDIR="features/train"

echo "[*] Training phase..."
python scripts/main_lightning.py --phase="train" \
    --train_file=${TRAINFILE} --fold_label_file=${FOLDLABELFILE} \
    --feats_dir=${FEATSDIR} --model_dir=${MODELDIR} --model_type=${MODELTYPE} \
    --loss_type=${LOSSTYPE} --centroids_file=${CENTFILE} \
    --input_dim=45 --channel_dims=${CHDIMS} --kernel_sizes="5_5" \
    --gru_dim=1024 --gru_bidirec=${BIDIREC} --hidden_dims="512" \
    --drop_prob=0.2 --activation_last=${ACTIV} --batch_norm=False \
    --batch_size_class=64 --loss_margin=0.${MARGIN} --loss_scale=${SCALE} \
    --ndata_workers=${NWORKERS}


###############################################################################
# Test phase
elif [[ $PHASE == "test" ]]; then

# Extract embeddings
TESTSET="lindahl"
TESTFILE="data/${TESTSET}/${TESTSET}.list"
if [[ $TESTSET == "lindahl" ]]; then SEP="_"; else SEP="."; fi
FEATSDIRTEST="features/${TESTSET}"
CKPTFILE="${MODELDIR}/checkpoint/model_epoch80.ckpt"

echo "[*] Extracting ${TESTSET^^} embeddings..."
python scripts/main_lightning.py --phase="extract" \
    --test_file=${TESTFILE} --scop_separation=${SEP} \
    --feats_dir_test=${FEATSDIRTEST} --model_dir=${MODELDIR} \
    --model_file=${CKPTFILE} --model_type=${MODELTYPE} \
    --loss_type=${LOSSTYPE} --centroids_file=${CENTFILE} \
    --input_dim=45 --channel_dims=${CHDIMS} --kernel_sizes="5_5" \
    --gru_dim=1024 --gru_bidirec=${BIDIREC} --hidden_dims="512" \
    --drop_prob=0.2 --activation_last=${ACTIV} --batch_norm=False \
    --ndata_workers=${NWORKERS}

# Compute cosine similarity scores and evaluate
EMBEDFILE="${MODELDIR}/embeddings/${TESTSET}.pkl"
SCORESDIR="${MODELDIR}/scores"

echo "[*] Computing cosine similarity scores and evaluating..."
./Run_eval_pairs_cosine.sh ${TESTSET} ${EMBEDFILE} ${SCORESDIR}

# Evaluate fold-level LINDAHL subset (2-cv)
if [[ $TESTSET == "lindahl" ]]; then
    echo "[*] Evaluating the fold-level LINDAHL subset (2-cv)..."
    python scripts/eval/eval_lindahl_fold_2sets.py ${SCORESDIR}/lindahl.score
fi

fi
