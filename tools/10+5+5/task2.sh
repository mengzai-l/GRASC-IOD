#!/bin/bash

set -e

PYTHON="python"

CONFIG_FILE="configs/faster-rcnn-increase/voc/10-5-5/grasc-iod/15+5_GR.py"
MAIN="tools/10+5+5/voc_15_5_main.py"

${PYTHON} ${MAIN} \

CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh ${CONFIG_FILE} 1