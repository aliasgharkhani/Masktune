#!/bin/bash
cd path/to/the/cloned/folder

python3 -m src.main --dataset catsvsdogs \
                    --train \
                    --arch resnet32  \
                    --base_dir path/to/the/cloned/folder \
                    --lr 0.001 \
                    --momentum 0.9 \
                    --weight_decay 1e-4 \
                    --gamma 0.1 \
                    --schedule 50 \
                    --use_cuda \
                    --optimizer adam \
                    --train_batch 128 \
                    --test_batch 128 \
                    --masking_batch 128 \
                    --epochs 200 \
                    --masktune \
                    --selective_classification \
                    --dataset_dir path/to/the/cloned/folder/ + datasets/catsvsdogs/raw
