#!/bin/bash
cd path/to/the/cloned/folder

python3 -m src.main --dataset celeba \
                    --train \
                    --arch resnet50 \
                    --base_dir path/to/the/cloned/folder \
                    --lr 0.0001 \
                    --use_cuda \
                    --optimizer sgd \
                    --train_batch 512 \
                    --test_batch 512 \
                    --masking_batch 512 \
                    --epochs 20 \
                    --gamma 1.0 \
                    --weight_decay 0.0001 \
                    --masktune \
                    --use_pretrained_weights \
                    --dataset_dir path/to/the/cloned/folder/ + datasets/CelebA/raw
