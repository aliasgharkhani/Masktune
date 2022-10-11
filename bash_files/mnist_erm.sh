#!/bin/bash
cd path/to/the/cloned/folder

python3 -m src.main --dataset mnist \
                    --train \
                    --arch small_cnn \
                    --base_dir path/to/the/cloned/folder \
                    --lr 0.01 \
                    --use_cuda \
                    --optimizer sgd \
                    --train_batch 128 \
                    --test_batch 128 \
                    --epochs 100 \
                    --train_bias_conflicting_data_ratio 0.01 \
                    --test_bias_conflicting_data_ratio 1.0 \
                    --bias_type square \
                    --test_data_type square \