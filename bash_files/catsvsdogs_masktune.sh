#!/bin/bash
cd path/to/the/cloned/folder

python3 -m src.main --dataset catsvsdogs \
                    --train \
                    --arch vgg16_bn  \
                    --base_dir path/to/the/cloned/folder \
                    --lr 0.1 \
                    --momentum 0.9 \
                    --weight_decay 5e-4 \
                    --gamma 0.5 \
                    --schedule 25 50 75 100 125 150 175 200 225 250 275 \
                    --use_cuda \
                    --optimizer sgd \
                    --train_batch 256 \
                    --test_batch 256 \
                    --masking_batch 256 \
                    --epochs 300 \
                    --masktune \
                    --selective_classification \
                    --dataset_dir path/to/the/cloned/folder/ + datasets/catsvsdogs/raw
