#!/bin/bash

set -e

PYTHON="python"

CONFIG_FILE="configs/faster-rcnn-increase/voc/10-10/grasc-iod/10+10_GR.py"
MAIN="tools/10+10/voc_10_10_main.py"

${PYTHON} ${MAIN} \

CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh ${CONFIG_FILE} 1

