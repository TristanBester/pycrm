#!/bin/sh
echo "Setting up environment variables..."
set -o allexport
source ../../.env
set +o allexport
echo -e "Setting up environment variables... DONE\n"

PROJECT_NAME=warehouse
PROJECT_PATH=../../
OUTPUT_PATH=../../experiments/warehouse/results/cf

# Make sure the output path exists
mkdir -p $OUTPUT_PATH

echo "Downloading logs from remote server..."
TARGET_DIR=experiments/warehouse/exp/cf/train/logs
rsync -avz -e ssh \
    $CLUSTER_USER@$CLUSTER_HOST:~/$PROJECT_NAME/$TARGET_DIR \
    $OUTPUT_PATH