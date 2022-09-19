This is an official pytorch implementation of [<em>MaskTune: Mitigating Spurious Correlations by Forcing to Explore</em>](https://duckduckgo.com), a NeurIPS 2022 paper. This work provides a technique for mitigating shortcut learning in machine learning algorithms.


1. clone the code (now you should have a folder named MaskTune)
2. Inside `MaskTune/` create `datasets/` folder
3. In `MaskTune/datasets/` create `catsvsdogs/raw/`, `CelebA/raw/`, `in9l/raw/`, and `Waterbirds/raw/` folders. (For other datasets (i.e., cifar10, mnist, and svhn) you don't need to do anything.)
4. To run an experiment, use bash files which are in `MaskTune/bash_files`. First of all, change the second line of the bash file to the path of `MaskTune` folder (e.g., `downloads/MaskTune`). You have to set `base_dir` to the path of `MaskTune/` folder and `dataset_dir` to the path of corresponding dataset (e.g., for celebA set this to `{base_dir}/datasets/CelebA/raw`)