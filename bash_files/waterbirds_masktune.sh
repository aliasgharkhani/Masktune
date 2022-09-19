#!/bin/bash
cd path/to/the/cloned/folder

python3 -m src.main --dataset waterbirds \
                    --train \
                    --arch resnet50 \
                    --base_dir path/to/the/cloned/folder \
                    --lr 0.0001 \
                    --use_cuda \
                    --optimizer sgd \
                    --train_batch 128 \
                    --test_batch 128 \
                    --masking_batch 128 \
                    --epochs 100 \
                    --masktune \
                    --dataset_dir path/to/the/cloned/folder/ + datasets/Waterbirds/raw
