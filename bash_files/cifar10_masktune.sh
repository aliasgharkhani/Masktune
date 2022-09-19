#!/bin/bash
cd path/to/the/cloned/folder

python3 -m src.main --dataset cifar10 \
                    --train \
                    --arch vgg16_bn  \
                    --base_dir path/to/the/cloned/folder \
                    --lr 0.1 \
                    --momentum 0.9 \
                    --weight_decay 5e-4 \
                    --gamma 0.5 \
                    --schedule 50 100 150 200 250 \
                    --use_cuda \
                    --optimizer sgd \
                    --train_batch 128 \
                    --test_batch 128 \
                    --masking_batch 128 \
                    --epochs 300 \
                    --selective_classification \
                    --masktune 