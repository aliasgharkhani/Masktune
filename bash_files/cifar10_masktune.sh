#!/bin/bash
cd path/to/the/cloned/folder

python3 -m src.main --dataset cifar10 \
                    --train \
                    --arch resnet32  \
                    --base_dir path/to/the/cloned/folder \
                    --lr 0.1 \
                    --momentum 0.9 \
                    --weight_decay 1e-4 \
                    --gamma 0.5 \
                    --schedule 25 50 75 100 125 150 175 200 225 250 275 \
                    --use_cuda \
                    --optimizer sgd \
                    --train_batch 128 \
                    --test_batch 128 \
                    --masking_batch 128 \
                    --epochs 300 \
                    --selective_classification \
                    --masktune 
