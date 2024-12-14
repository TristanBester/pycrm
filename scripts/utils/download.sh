#!/bin/sh
echo "Setting up environment variables..."
set -o allexport
source ../../../../.env
set +o allexport
echo -e "Setting up environment variables... DONE\n"

OUTPUT_PATH=../../checkpoints

echo "Downloading logs from remote server..."
rsync -avz -e ssh \
    $CLUSTER_USER@$CLUSTER_HOST:/datasets/tbester/warehouse/C-SAC_joints_p-1_1 \
    $OUTPUT_PATH
