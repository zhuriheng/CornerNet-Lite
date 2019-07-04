#!/usr/bin/env bash
set -eux
set -o pipefail

source activate CornerNet_Lite

#nohup python -u train.py CornerNet_Saccade_voc > CornerNet_Saccade_voc.log 2>&1 &

nohup python -u train.py CornerNet_Saccade_terrorpost --workers 4 > CornerNet_Saccade_terrorpost.log 2>&1 &