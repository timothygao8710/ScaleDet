# ScaleMAE-Det

This is a minimal implementation of [VitDET](https://arxiv.org/abs/2203.16527) with a [ScaleMAE](https://arxiv.org/abs/2212.14532) Backbone. We utilize a simple feature pyramid network as highlighted in the VitDET paper, and RCNN detection head.

ScaleMAE trains on (image, scale) tuples, encodes scale into positional embedding, and can learn generalize to scales outside of training distribution. This is particularly useful for satellite imagery. We utilize the pretrained ScaleMAE backbone, through away the classification head and token, and find it achieves up to 0.236 mAP on the xView Dataset.

* This repo is a modification on the [ScaleMAE repo](https://github.com/bair-climate-initiative/scale-mae)

* As mentioned in the MAE repo, this repo is based on [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+. In addition, install gdal, rasterio, and Shapely.  This tends to work pretty well (but gdal is notoriously tricky):

## Installation
```bash
conda create -n scalemae python=3.9 geopandas # geopandas should install gdal correctly
conda activate scalemae
# replace with your desired pytorch target (e.g. cuda version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -e .
```

## Pretrained ScaleMAE

* [**ViT Large 800 ep**](https://github.com/bair-climate-initiative/scale-mae/releases/download/base-800/scalemae-vitlarge-800.pth)

### Finetuning

```
python -m torch.distributed.launch --nproc_per_node=4 train.py \
    --train_image_dir /path/to/train/images \
    --train_label_file /path/to/train/labels.json \
    --val_image_dir /path/to/val/images \
    --val_label_file /path/to/val/labels.json \
    --batch_size 1 \
    --learning_rate 5e-5 \
    --num_epochs 40 \
    --input_size 800 \
    --save_checkpoint_path /path/to/save/checkpoints/checkpoint.pth
```
