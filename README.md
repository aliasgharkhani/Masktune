training the baseline for mnist:
```
python3 -m src.main --dataset mnist --train --arch small_cnn --base_dir /localhome/aka225/Research/AugMask_Project/AugMask_v1 --lr 0.00001 --use_cuda --square_percent 0.8 --log_images
```

running iterative masking with mean masking:

```
python3 -m src.main --dataset mnist --train --arch small_cnn --base_dir /localhome/aka225/Research/AugMask_Project/AugMask_v1 --lr 0.00001 --use_cuda --square_percent 0.8 --iterative_masking --masking mean_mask --log_images
```

running iterative masking with sort masking:

```
python3 -m src.main --dataset mnist --train --arch small_cnn --base_dir /localhome/aka225/Research/AugMask_Project/AugMask_v1 --lr 0.00001 --use_cuda --square_percent 0.8 --iterative_masking --masking mean_mask --log_images --remove_k 100
```