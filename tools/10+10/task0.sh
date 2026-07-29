#!/bin/bash

set -e

PYTHON="python"
CONFIG_FILE="configs/faster-rcnn/voc/10/10.py"

CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh ${CONFIG_FILE} 1

