import os
import sys
import time
import wandb
import torch
import torch.distributed as dist
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import utils
from dataset import COCOFormatDataset
# from engine import train_one_epoch, evaluate
from lightning.fabric import Fabric
from vitdet import get_object_detection_model
from torchvision.models.vision_transformer import EncoderBlock
from functools import partial
from lightning.fabric.strategies import FSDPStrategy


class Configs:
    def __init__(self):
        self.project_name = "geospatial-transition"
        self.entity_name = "timothy-gao"
        self.num_classes = 60
        
        self.train_image_dir = "/home/timothygao/xview_400_0/train_images_400_0"
        self.train_label_file = "/home/timothygao/xview_400_0/train_full_labels.json"
        self.val_image_dir = "/home/timothygao/xview_400_0/val_images_400_0"
        self.val_label_file = "/home/timothygao/xview_400_0/val_full_labels.json"

        # self.train_image_dir = "/datasets/coco2017_2024-01-05_0047/train2017"
        # self.train_label_file = "/datasets/coco2017_2024-01-05_0047/annotations/instances_train2017.json"
        # self.val_image_dir = "/datasets/coco2017_2024-01-05_0047/val2017"
        # self.val_label_file = "/datasets/coco2017_2024-01-05_0047/annotations/instances_val2017.json"

        self.batch_size = 1
        self.num_workers = 2
        self.learning_rate = 5e-5
        self.weight_decay = 0.0005
        self.num_epochs = 40
        self.print_freq = 20
        self.input_size = 800

        # self.load_checkpoint_path = f"/home/timothygao/scalemae_docker/checkpoints/latest_checkpoint_{self.input_size}.pth"
        self.load_checkpoint_path = None
        self.save_checkpoint_path = f"/home/timothygao/scalemae_docker/checkpoints/latest_checkpoint_800_test.pth"


def get_transform(train, config):
    transforms = []
    transforms.append(T.Resize([config.input_size, config.input_size])) #FIXME: Modifies scale?

    # transforms.append(T.Normalize(mean=[0.23049139083515527, 0.19105629050902187, 0.1564653866611124], 
    #                               std=[0.17102769749821609, 0.13835342196803596, 0.12774692102103344]))

    if train:
        transforms.extend([
            T.RandomHorizontalFlip(0.5),

            T.RandomApply(torch.nn.ModuleList([  
                T.RandomChoice( # Look at the shape, not color
                [
                    T.RandomGrayscale(p=0.5),
                    T.RandomSolarize(threshold=192.0, p=0.5)
                ])
            ]), p=0.7),

            T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5),

            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)
            ]
        )

    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    # transforms.append(T.Normalize(mean=[0.23049139083515527, 0.19105629050902187, 0.1564653866611124], 
    #                               std=[0.17102769749821609, 0.13835342196803596, 0.12774692102103344]))

    return T.Compose(transforms)

def save_checkpoint(model, optimizer, epoch, config, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename, map_location='cpu')
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
import wandb
import torch.distributed as dist


def train_one_epoch(fabric, model, optimizer, data_loader, epoch, print_freq, rank):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    all_loss_sums = []
    window_len = 100
    print(torch.cuda.memory_summary())
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = fabric.to_device(images)
        targets = [{k: fabric.to_device(v) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]


        print(torch.cuda.memory_summary())
        with fabric.autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        print(torch.cuda.memory_summary())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        print(torch.cuda.memory_summary())
        loss_value = losses_reduced.item()
        all_loss_sums.append(loss_value)

        if rank == 0:
            wandb.log(loss_dict_reduced)
            mn = min(len(all_loss_sums), window_len)
            wandb.log({
                "sum_loss": loss_value,
                "smooth_loss": sum(all_loss_sums[-mn:]) / mn
            })

        print(torch.cuda.memory_summary())
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        fabric.backward(losses)
        optimizer.step()
        
        print(torch.cuda.memory_summary())

        if lr_scheduler is not None:
            lr_scheduler.step()

        if rank == 0:
            wandb.log({"lr": optimizer.param_groups[0]["lr"]})
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

def main():
    config = Configs()
    
    # auto_wrap_policy = partial(transformer_auto_wrap_policy,
    #     transformer_layer_cls={EncoderBlock}
    # )
    # strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy)
    fabric = Fabric(accelerator="cuda", 
        strategy="fsdp", precision="16-mixed"
    )
    fabric.launch()
    
    if fabric.global_rank == 0:
        wandb.init(project=config.project_name, entity=config.entity_name)
    fabric.barrier()
    
    dataset = COCOFormatDataset(config.train_image_dir,
                                config.train_label_file,
                                get_transform(train=True, config=config))
    
    dataset_test = COCOFormatDataset(config.val_image_dir,
                                     config.val_label_file,
                                     get_transform(train=False, config=config))
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=utils.collate_fn,
        shuffle=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=utils.collate_fn,
        shuffle=True
    )
    
    model = get_object_detection_model(input_size=config.input_size, num_classes=config.num_classes)    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    
    model, optimizer = fabric.setup(model, optimizer)
    data_loader, data_loader_test = fabric.setup_dataloaders(data_loader, data_loader_test)
    
    if config.load_checkpoint_path:
        load_checkpoint(model, optimizer, config.load_checkpoint_path)
        print(f"Resuming training from epoch {config.load_checkpoint_path}")
    
    for epoch in range(0, config.num_epochs):
        train_one_epoch(fabric, model, optimizer, data_loader, epoch, print_freq=config.print_freq, rank=fabric.global_rank)
        
        if fabric.global_rank == 0:
            save_checkpoint(model, optimizer, epoch + 1, config, config.save_checkpoint_path)
            print(f"Checkpoint saved to {config.save_checkpoint_path}")

        fabric.barrier()

    print('Training completed!')

if __name__ == "__main__":
    main()