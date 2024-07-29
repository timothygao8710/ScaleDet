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
import utils
from dataset import COCOFormatDataset
from engine import train_one_epoch, evaluate
from lightning.fabric import Fabric


from vitdet import get_object_detection_model

# def get_object_detection_model(input_size, num_classes):
#     weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
#     model = fasterrcnn_resnet50_fpn_v2(weights=weights)
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#     return model

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
        self.save_checkpoint_path = f"/home/timothygao/scalemae_docker/checkpoints/latest_checkpoint_800_full_vit.pth"


def get_transform(train, config):
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

# def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler):
#     model.train()
#     for images, targets in data_loader:
#         images = list(img.to(device) for img in images)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
#         with torch.cuda.amp.autocast():
#             loss_dict = model(images, targets)
#             losses = sum(loss for loss in loss_dict.values())
        
#         optimizer.zero_grad()
#         scaler.scale(losses).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         if print_freq > 0:
#             print(f"Epoch: [{epoch}], Loss: {losses.item()}")

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
        batch_size=1,  # Lower batch size
        sampler=train_sampler,
        num_workers=2,  # Reduce the number of workers
        pin_memory=True,  # Use pin_memory
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,  # Lower batch size
        sampler=test_sampler,
        num_workers=2,  # Reduce the number of workers
        pin_memory=True,  # Use pin_memory
        collate_fn=utils.collate_fn
    )
    
    model = get_object_detection_model(input_size=400, num_classes=60)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    scaler = torch.cuda.amp.GradScaler()
    
    num_epochs = 10
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10, scaler=scaler)
        lr_scheduler.step()
        
        # Evaluate on the test dataset
        coco_evaluator = evaluate(model, data_loader_test, device=device)
        
        if config.save_checkpoint_path and rank == 0:
            # save_checkpoint(model, optimizer, epoch + 1, config, 
            #                 os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            save_checkpoint(model, optimizer, epoch + 1, config, config.save_checkpoint_path)
            print(f"Checkpoint saved to {config.save_checkpoint_path}")

        if rank == 0:
            wandb.log({
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "mAP": coco_evaluator.coco_eval["bbox"].stats[0],
                "mAP_50": coco_evaluator.coco_eval["bbox"].stats[1],
            })
        
        if epoch % 2 == 0 and rank == 0:  # Log every 2 epochs, only on rank 0
            log_sample_predictions(model, data_loader_test, device, num_images=10, epoch=epoch)

    print('=' * 100)
    print('FINAL EVAL ' * 10)
    print('=' * 100)

    if rank == 0:
        wandb.finish()
    print("Training completed!")

def log_sample_predictions(model, data_loader, device, num_images=5, epoch=0, save_dir='sample_predictions'):
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
            
            # Save images locally
            pred_pil = Image.fromarray(pred_img.permute(1, 2, 0).numpy())
            gt_pil = Image.fromarray(gt_img.permute(1, 2, 0).numpy())
            pred_pil.save(os.path.join(save_dir, f"prediction_{images_logged}_{epoch}.png"))
            gt_pil.save(os.path.join(save_dir, f"ground_truth_{images_logged}_{epoch}.png"))

            print(os.path.join(save_dir, f"prediction_{images_logged}_{epoch}.png"))
            print(os.path.join(save_dir, f"ground_truth_{images_logged}_{epoch}.png"))
            
            images_logged += 1
            if images_logged >= num_images:
                return

if __name__ == "__main__":
    main()


# torchrun --nproc_per_node=10 train_mem.py
# torchrun --nproc_per_node=8 train_mem.py
# torchrun --nproc_per_node=5 train_mem.py
# torchrun --nproc_per_node=1 train_mem.py
