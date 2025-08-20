#!/bin/sh
echo "Setting up environment variables..."
set -o allexport
source ../../.env
set +o allexport
echo -e "Setting up environment variables... DONE\n"


OUTPUT_PATH="../../results/rm/"
rsync -avz -e ssh \
"$CLUSTER_USER@$CLUSTER_HOST:/home-mscluster/tbester/warehouse/examples/rm/discrete/logs" \
"$OUTPUT_PATH"



