#!/bin/sh
echo "Setting up environment variables..."
set -o allexport
source ../../.env
set +o allexport
echo -e "Setting up environment variables... DONE\n"

PROJECT_NAME=warehouse
PROJECT_PATH=../../

echo "Copying project files to remote server..."
rsync -avz -e ssh \
    --exclude="*.pyc" \
    --exclude="*.ipynb_checkpoints" \
    --exclude=".env" \
    --exclude="*.git" \
    --exclude="*.tfevents*" \
    --exclude="*.gitignore" \
    --exclude="*.DS_Store" \
    --exclude=".venv*" \
    --exclude="wandb" \
    --exclude="__pycache__" \
    --exclude="results" \
    --exclude="*.zip" \
    --exclude=".ruff_cache" \
    --exclude="logs" \
    --exclude="videos" \
    --exclude="archive" \
    --exclude=".tox" \
    $PROJECT_PATH  \
    $CLUSTER_USER@$CLUSTER_HOST:~/$PROJECT_NAME