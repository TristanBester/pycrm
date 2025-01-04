#!/bin/sh
echo "Setting up environment variables..."
set -o allexport
source ../../../../.env
set +o allexport
echo -e "Setting up environment variables... DONE\n"

# Define step counts array using proper bash syntax
STEP_COUNTS=( 
5010000
5040000
5070000
5100000
5130000
5160000
5190000
5220000
5250000
5280000
5310000
)

# Loop through experiments and step counts
for i in {0..19}; do
    for j in "${STEP_COUNTS[@]}"; do
        echo "Downloading model_${i}_${j}.zip"
        OUTPUT_PATH="../../checkpoints_old/model_${i}_${j}.zip"
        rsync -avz -e ssh \
        "$CLUSTER_USER@$CLUSTER_HOST:/datasets/tbester/v1-checkpoints/cs-checkpoints/CSAC_default_${i}/model_${j}_steps.zip" \
        "$OUTPUT_PATH"
    done
done




