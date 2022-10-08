<h1>MaskTune: Mitigating Spurious Correlations by Forcing to Explore, NeurIPS 2022</h1>
This is the official pytorch implementation of <a href="http://arxiv.org/abs/2210.00055"><em>MaskTune: Mitigating Spurious Correlations by Forcing to Explore</em></a>, <strong>NeurIPS 2022</strong>. MaskTune is a technique for mitigating shortcut learning in machine learning algorithms.

<br>
</br>

<div align="center">
<img src="https://github.com/aliasgharkhani/Masktune/blob/master/masktune_method_.png" width="800" height="280"">
</div>


<h1>How to use</h1>

1. Clone the code (now you should have a folder named MaskTune)
2. Inside `Masktune/` create `datasets/` folder
3. For `catsvsdogs` and `inl9 (the Background Challenge)` expriments, inside `MaskTune/datasets/` create `catsvsdogs/raw/` and `in9l/raw/` folders. For other datasets ignore this step.
4. Download the dataset you want (you don't need to download cifar10, mnist, and svhn because they will be downloaded automatically).
5. For Waterbirds, please download the <strong>corrected version of the Waterbirds dataset</strong> from <a href="https://drive.google.com/file/d/1xPNYQskEXuPhuqT5Hj4hXPeJa9jh7liL/view?usp=sharing">here</a> (The original Waterbirds dataset has some label and image noise). Then extract it into the `Masktune/datasets/Waterbirds/` (so inside this folder you should have `images` folder)
6. For CelebA, please download the dataset from <a href="https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?resource=download-directory">here</a>. After extracting it, you should see a folder named `archive`. Pass this folder's path to the `--dataset_dir` in the bash file.
7. To run an experiment, use the bash files in `MaskTune/bash_files`. First, change the second line of the bash file to the path of `MaskTune` folder (e.g., `downloads/MaskTune`). You have to set `base_dir` to the path of `MaskTune/` folder and `dataset_dir` to the path of corresponding dataset (e.g., for celebA set this to `{base_dir}/datasets/CelebA/raw`)
