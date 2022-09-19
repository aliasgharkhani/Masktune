#!/bin/bash
cd /localhome/aka225/Research/AugMask_Project/MaskTune/AugMask_v1
source ../../AugMask_v1/augmask_venv/bin/activate

python3 -m src.main --dataset mnist \
                    --train \
                    --arch small_cnn \
                    --base_dir /localhome/aka225/Research/AugMask_Project/MaskTune/AugMask_v1 \
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