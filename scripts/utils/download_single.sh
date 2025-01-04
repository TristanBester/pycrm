#!/bin/sh
echo "Setting up environment variables..."
set -o allexport
source ../../../../.env
set +o allexport
echo -e "Setting up environment variables... DONE\n"


OUTPUT_PATH="../../model.zip"
rsync -avz -e ssh \
"$CLUSTER_USER@$CLUSTER_HOST:/datasets/tbester/v1-checkpoints/cs-checkpoints/CSAC_default_0/model_5100000_steps.zip" \
"$OUTPUT_PATH"



