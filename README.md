# Scale-MAE 🛰️

![image](https://user-images.githubusercontent.com/1455579/217665789-b46d6830-445f-4151-b7a4-a2152a81a8d1.png)


This repository provides a reimplementation of the code for [Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning](https://arxiv.org/abs/2212.14532), integrated with VitDet.

```
@article{reed2022scale,
  title={Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning},
  author={Reed, Colorado J and Gupta, Ritwik and Li, Shufan and Brockman, Sarah and Funk, Christopher and Clipp, Brian and Candido, Salvatore and Uyttendaele, Matt and Darrell, Trevor},
  journal={arXiv preprint arXiv:2212.14532},
  year={2022}
}
```

* This repo is a modification on the [ScaleMAE repo]([https://github.com/facebookresearch/mae](https://github.com/bair-climate-initiative/scale-mae)). Installation and preparation follow that repo ;-).

* As mentioned in the MAE repo, this repo is based on [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+. In addition, install gdal, rasterio, and Shapely.  This tends to work pretty well (but gdal is notoriously tricky):

## Installation
```bash
conda create -n scalemae python=3.9 geopandas # geopandas should install gdal correctly
conda activate scalemae
# replace with your desired pytorch target (e.g. cuda version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -e .
```

## Data Preparation
Download the FMoW-rgb dataset as described in the [here](https://github.com/fMoW/dataset) and then make a symlink to the data directory in the root of this repo.  For example, if you downloaded the data to `~/data/fmow-rgb`, then run:

```bash
ln -s ~/data/fmow-rgb data
```

## Pretraining ##
Datasets are defined by config files in `config`.
```
# change to num of gpus you have
python -m torch.distributed.launch --nproc_per_node=4
main_pretrain.py
```

use `-h` to see details of all arguments. 


## Pretrained Models

* [**ViT Large 800 ep**](https://github.com/bair-climate-initiative/scale-mae/releases/download/base-800/scalemae-vitlarge-800.pth)


## Evaluation

### Finetuning

```
python -m torch.distributed.launch --nproc_per_node=4 \
train_mem.py \
--checkpoint_path <path-to-model-checkpoint.pth>
```
