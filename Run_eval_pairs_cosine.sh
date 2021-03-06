#!/bin/bash


DATASET=$1
EMBEDFIL=$2
OUTDIR=$3

# Get scoring for each pair of proteins
NAMESFIL=data/${DATASET}/level_pairs/${DATASET}_names
SCOREFIL=${OUTDIR}/${DATASET}.score
mkdir -p ${OUTDIR}

if [[ $EMBEDFIL != "none" ]]; then    # get scores if provided embedding file
    python scripts/eval/scoring.py ${NAMESFIL} ${EMBEDFIL} ${SCOREFIL}
fi

FAMFIL=data/${DATASET}/level_pairs/${DATASET}_family
SUPFAMFIL=data/${DATASET}/level_pairs/${DATASET}_superfamily
FOLDFIL=data/${DATASET}/level_pairs/${DATASET}_fold

# Get Top1 / Top5 accuracy predictions
RESFIL=${OUTDIR}/${DATASET}_results.txt
TMPDIR=${OUTDIR}/TMP
mkdir -p ${TMPDIR}

printf "Calculating correctly predicted template at family level (Top1, Top5): \n" > ${RESFIL}
python scripts/eval/calculate_top1_top5.py ${SCOREFIL} ${FAMFIL} >> ${RESFIL}
printf "\nCalculating correctly predicted template at superfamily level (Top1, Top5): \n" >> ${RESFIL}
grep -F -v -f  ${FAMFIL} ${SCOREFIL} > ${TMPDIR}/deleted-fam
python scripts/eval/calculate_top1_top5.py ${TMPDIR}/deleted-fam ${SUPFAMFIL} >> ${RESFIL}
printf "\nCalculating correctly predicted template at fold level (Top1, Top5): \n" >> ${RESFIL}
grep -F -v -f ${SUPFAMFIL} ${TMPDIR}/deleted-fam > ${TMPDIR}/deleted-fam-supfam
python scripts/eval/calculate_top1_top5.py ${TMPDIR}/deleted-fam-supfam ${FOLDFIL} >> ${RESFIL}
printf "" >> ${RESFIL}

rm -r $TMPDIR
