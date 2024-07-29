import os
import math
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
import utils
from dataset import COCOFormatDataset
from engine import train_one_epoch, evaluate
# from vitdet import get_object_detection_model

def get_object_detection_model(input_size, num_classes):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    setup_for_distributed(rank == 0)

def main():
    init_distributed_mode()
    rank = dist.get_rank()
    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        wandb.init(project="geospatial-transition", entity="timothy-gao")
    
    num_classes = 60
    dataset = COCOFormatDataset(get_transform(train=True))
    dataset_test = COCOFormatDataset(get_transform(train=False))
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,  # This is the per-GPU batch size
        sampler=train_sampler,
        num_workers=4,
        collate_fn=utils.collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        sampler=test_sampler,
        num_workers=4,
        collate_fn=utils.collate_fn
    )
    
    model = get_object_detection_model(input_size=400, num_classes=60)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        
        # Evaluate on the test dataset
        coco_evaluator = evaluate(model, data_loader_test, device=device)
        
        if rank == 0:
            wandb.log({
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "mAP": coco_evaluator.coco_eval["bbox"].stats[0],
                "mAP_50": coco_evaluator.coco_eval["bbox"].stats[1],
            })
        
        if epoch % 2 == 0 and rank == 0:  # Log every 2 epochs, only on rank 0
            log_sample_predictions(model, data_loader_test, device, num_images=5)
    
    if rank == 0:
        wandb.finish()
    print("Training completed!")

def log_sample_predictions(model, data_loader, device, num_images=5):
    model.eval()
    images_logged = 0
    
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        
        with torch.no_grad():
            predictions = model(images)
        
        for img, pred, target in zip(images, predictions, targets):
            if images_logged >= num_images:
                return
            
            # Convert image to uint8 and move to CPU
            img_uint8 = (img * 255).byte().cpu()
            
            # Draw predicted boxes
            pred_boxes = pred['boxes'].cpu()
            pred_labels = pred['labels'].cpu()
            pred_scores = pred['scores'].cpu()
            threshold = 0.7
            keep = pred_scores > threshold
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            pred_scores = pred_scores[keep]
            
            pred_img = draw_bounding_boxes(img_uint8, pred_boxes, labels=[f"{l.item()}: {s:.2f}" for l, s in zip(pred_labels, pred_scores)], colors="red", width=2)
            
            # Draw ground truth boxes
            gt_boxes = target['boxes'].cpu()
            gt_labels = target['labels'].cpu()
            gt_img = draw_bounding_boxes(img_uint8, gt_boxes, labels=[f"{l.item()}" for l in gt_labels], colors="green", width=2)
            
            # Log images to wandb
            wandb.log({
                f"predictions_{images_logged}": wandb.Image(pred_img.permute(1, 2, 0).numpy()),
                f"ground_truth_{images_logged}": wandb.Image(gt_img.permute(1, 2, 0).numpy())
            })
            
            images_logged += 1
            if images_logged >= num_images:
                return

if __name__ == "__main__":
    main()

# torchrun --nproc_per_node=8 train.py
# torchrun --nproc_per_node=1 train.py