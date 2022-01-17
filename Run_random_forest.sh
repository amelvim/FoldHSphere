#!/bin/bash


DATASET=$1
# lindahl / lindahl_1.75 / lindahl_cv2
NJOBS=$2

RESDIR="results"
OUTDIR="${RESDIR}/FoldHSpherePro"
mkdir -p ${OUTDIR}

# LINDAHL and LINDAHL_1.75
if [[ $DATASET == "lindahl" || $DATASET == "lindahl_1.75" ]]; then
    SIMFILE="${RESDIR}/similarity_measures/${DATASET}_sim.pkl"
    SCORESFILES="${RESDIR}/FoldHSphere/${DATASET}.score"
    SCORESFILES+="+${RESDIR}/DeepFR_s2/${DATASET}.score"

    for PART in {1..10}; do
        TRAINFILE="data/${DATASET}/cv10_pairs/train${PART}.pairs"
        TESTFILE="data/${DATASET}/cv10_pairs/test${PART}.pairs"
        SAVEFILE="${OUTDIR}/${DATASET}_${PART}.score"

        python scripts/random-forest/rf_lindahl.py \
            $SIMFILE $SCORESFILES $TRAINFILE $TESTFILE $SAVEFILE $NJOBS
    done
    cat ${OUTDIR}/${DATASET}_{1..10}.score > ${OUTDIR}/${DATASET}.score

    echo "[*] Evaluating..."
    ./Run_eval_pairs_cosine.sh ${DATASET} "none" ${OUTDIR}

# LINDAHL cv2
elif [[ $DATASET == "lindahl_cv2" ]]; then
    SIMFILE="${RESDIR}/similarity_measures/lindahl_sim.pkl"
    SCORESFILES="${RESDIR}/FoldHSphere/lindahl.score"
    SCORESFILES+="+${RESDIR}/DeepFR_s2/lindahl.score"

    TRAINFILE="data/lindahl/cv2_fold_level/LE_2_2.pairs"
    TESTFILE="data/lindahl/cv2_fold_level/LE_1_2.pairs"
    SAVEFILE="${OUTDIR}/lindahl_cv2_1_2.score"
    python scripts/random-forest/rf_lindahl.py \
        $SIMFILE $SCORESFILES $TRAINFILE $TESTFILE $SAVEFILE $NJOBS

    TRAINFILE="data/lindahl/cv2_fold_level/LE_1_1.pairs"
    TESTFILE="data/lindahl/cv2_fold_level/LE_2_1.pairs"
    SAVEFILE="${OUTDIR}/lindahl_cv2_2_1.score"
    python scripts/random-forest/rf_lindahl.py \
        $SIMFILE $SCORESFILES $TRAINFILE $TESTFILE $SAVEFILE $NJOBS

    cat ${OUTDIR}/lindahl_cv2_*.score > ${OUTDIR}/lindahl_cv2.score

    echo "[*] Evaluating..."
    python scripts/eval/eval_lindahl_fold_2sets.py ${OUTDIR}/lindahl_cv2.score

fi
