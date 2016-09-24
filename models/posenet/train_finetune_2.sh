#!/usr/bin/env sh

./build/tools/caffe train -solver models/posenet/solver_finetune_2.prototxt -weights models/posenet/posenet_pretrain.caffemodel -gpu 2,3,4,5,6 2>&1 | tee models/posenet/train_log_finetune_2.txt

