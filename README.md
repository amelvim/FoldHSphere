# FoldHSphere: Deep Hyperspherical Embeddings for Protein Fold Recognition

## Downloadable data

Input data, features and trained models can be found at <http://sigmat.ugr.es/~amelia/FoldHSphere/>.

## Run

### Neural network model

```bash
./Run_train_test.sh
```

1. Train the ResCNN-BGRU model using Thomson-LMCL approach:

```bash
TRAINFILE="data/train/train.list"
FOLDLABELFILE="data/train/fold_label_relation_1154.txt"
FEATSDIR="features/train"
MODELDIR="models/ResCNN-BGRU/tanh_thomson_lmcl_m06_s30"; mkdir -p $MODELDIR
CENTDIR="models/prototypes_thomson/optim_rescnn-bgru-softmax/thl_sum"

python scripts/main_lightning.py --phase="train" \
    --train_file=${TRAINFILE} --fold_label_file=${FOLDLABELFILE} \
    --feats_dir=${FEATSDIR} --model_dir=${MODELDIR} \
    --model_type="rescnn_gru" --loss_type="lmcl_fixed" \
    --centroids_file="${CENTDIR}/prototypes_ep1020.npy" \
    --input_dim=45 --channel_dims="64_256_64_256" --kernel_sizes="5_5" \
    --gru_dim=1024 --gru_bidirec=True --hidden_dims="512" \
    --drop_prob=0.2 --activation_last="tanh" --batch_norm=False \
    --batch_size_class=64 --loss_margin=0.6 --loss_scale=30 \
    --ndata_workers=2
```

2. Extract embeddings for the LINDAHL dataset using the ResCNN-BGRU pre-trained model:

```bash
TESTFILE="data/lindahl/lindahl.list"
FEATSDIRTEST="features/lindahl"

python scripts/main_lightning.py --phase="extract" \
    --test_file=${TESTFILE} --scop_separation="_" \
    --feats_dir_test=${FEATSDIRTEST} --model_dir=${MODELDIR} \
    --model_file="${MODELDIR}/checkpoint/model_epoch80.ckpt" \
    --model_type="rescnn_gru" --loss_type="lmcl_fixed" \
    --centroids_file="${CENTDIR}/prototypes_ep1020.npy" \
    --input_dim=45 --channel_dims="64_256_64_256" --kernel_sizes="5_5" \
    --gru_dim=1024 --gru_bidirec=True --hidden_dims="512" \
    --drop_prob=0.2 --activation_last="tanh" --batch_norm=False \
    --ndata_workers=2
```

3. Compute cosine similarity scores and evaluate:

```bash
EMBEDFILE="${MODELDIR}/embeddings/lindahl.pkl"
SCORESDIR="${MODELDIR}/scores"

./Run_eval_pairs_cosine.sh "lindahl" ${EMBEDFILE} ${SCORESDIR}
```

### Random forest model

```bash
./Run_random_forest.sh "lindahl" 4
```

## Requirements

- Python 3.7.7
- Numpy 1.19.0
- Scikit-Learn 0.23.1
- Matplotlib 3.2.2
- PyTorch 1.4.0
- Tensorboard 2.2.0
- PyTorch-Lightning 0.10.0
