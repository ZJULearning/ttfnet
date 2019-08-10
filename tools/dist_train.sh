#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS  \
    --master_port=$((RANDOM + 10000)) \
    $(dirname "$0")/train.py $CONFIG --validate --launcher pytorch ${@:3} --autoscale-lr --gpus=$GPUS
