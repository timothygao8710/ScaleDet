import os
import torch
import torch.distributed as dist
import torchvision.transforms.v2 as T
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import wandb
import argparse

from dataset import COCOFormatDataset
from engine import train_one_epoch, evaluate
from vitdet import get_object_detection_model
import utils

class Config:
    def __init__(self, args):
        self.project_name = args.project_name
        self.entity_name = args.entity_name
        self.num_classes = args.num_classes
        
        self.train_image_dir = args.train_image_dir
        self.train_label_file = args.train_label_file
        self.val_image_dir = args.val_image_dir
        self.val_label_file = args.val_label_file

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.num_epochs = args.num_epochs
        self.print_freq = args.print_freq
        self.input_size = args.input_size

        self.load_checkpoint_path = args.load_checkpoint_path
        self.save_checkpoint_path = args.save_checkpoint_path

def get_transform(train, config):
    transforms = [
        T.Resize([config.input_size, config.input_size]),
        T.ToDtype(torch.float, scale=True),
        T.ToPureTensor()
    ]
    
    if train:
        transforms.insert(1, T.RandomHorizontalFlip(0.5))
        transforms.insert(2, T.RandomApply(torch.nn.ModuleList([
            T.RandomChoice([
                T.RandomGrayscale(p=0.5),
                T.RandomSolarize(threshold=192.0, p=0.5)
            ])
        ]), p=0.7))
        transforms.insert(3, T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5))
        transforms.insert(4, T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2))

    return T.Compose(transforms)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename, map_location='cpu')
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def save_checkpoint(model, optimizer, epoch, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)

def setup_for_distributed(is_master):
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

def log_sample_predictions(model, data_loader, device, num_images=5, save_dir='sample_predictions'):
    model.eval()
    images_logged = 0
    
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        
        with torch.no_grad():
            predictions = model(images)
        
        for img, pred, target in zip(images, predictions, targets):
            if images_logged >= num_images:
                return
            
            img_uint8 = (img * 255).byte().cpu()
            
            pred_boxes = pred['boxes'].cpu()
            pred_labels = pred['labels'].cpu()
            pred_scores = pred['scores'].cpu()
            threshold = 0.7
            keep = pred_scores > threshold
            pred_boxes, pred_labels, pred_scores = pred_boxes[keep], pred_labels[keep], pred_scores[keep]
            
            pred_img = draw_bounding_boxes(img_uint8, pred_boxes, labels=[f"{l.item()}: {s:.2f}" for l, s in zip(pred_labels, pred_scores)], colors="red", width=2)
            
            gt_boxes = target['boxes'].cpu()
            gt_labels = target['labels'].cpu()
            gt_img = draw_bounding_boxes(img_uint8, gt_boxes, labels=[f"{l.item()}" for l in gt_labels], colors="green", width=2)
            
            wandb.log({
                f"predictions_{images_logged}": wandb.Image(pred_img.permute(1, 2, 0).numpy()),
                f"ground_truth_{images_logged}": wandb.Image(gt_img.permute(1, 2, 0).numpy())
            })
            
            pred_pil = Image.fromarray(pred_img.permute(1, 2, 0).numpy())
            gt_pil = Image.fromarray(gt_img.permute(1, 2, 0).numpy())
            pred_pil.save(os.path.join(save_dir, f"prediction_{images_logged}.png"))
            gt_pil.save(os.path.join(save_dir, f"ground_truth_{images_logged}.png"))

            print(f"Saved: {os.path.join(save_dir, f'prediction_{images_logged}.png')}")
            print(f"Saved: {os.path.join(save_dir, f'ground_truth_{images_logged}.png')}")
            
            images_logged += 1

    
def parse_args():
    parser = argparse.ArgumentParser(description="Object Detection Training")
    parser.add_argument("--project_name", type=str, default="scalemaedet", help="Project name for wandb")
    parser.add_argument("--entity_name", type=str, default="timothy-gao", help="Entity name for wandb")
    parser.add_argument("--num_classes", type=int, default=60, help="Number of classes")
    parser.add_argument("--train_image_dir", type=str, required=True, help="Path to training images")
    parser.add_argument("--train_label_file", type=str, required=True, help="Path to training labels")
    parser.add_argument("--val_image_dir", type=str, required=True, help="Path to validation images")
    parser.add_argument("--val_label_file", type=str, required=True, help="Path to validation labels")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loading workers")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=40, help="Number of epochs")
    parser.add_argument("--print_freq", type=int, default=20, help="Print frequency")
    parser.add_argument("--input_size", type=int, default=800, help="Input image size")
    parser.add_argument("--load_checkpoint_path", type=str, default=None, help="Path to load checkpoint from")
    parser.add_argument("--save_checkpoint_path", type=str, default="/home/timothygao/scalemae_docker/checkpoints/latest_checkpoint_800_full_vit.pth", help="Path to save checkpoints")
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config(args)
    init_distributed_mode()
    rank = dist.get_rank()
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        wandb.init(project=config.project_name, entity=config.entity_name)
    
    dataset = COCOFormatDataset(config.train_image_dir, config.train_label_file, get_transform(train=True, config=config))
    dataset_test = COCOFormatDataset(config.val_image_dir, config.val_label_file, get_transform(train=False, config=config))
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, sampler=train_sampler,
        num_workers=config.num_workers, pin_memory=True, collate_fn=utils.collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=config.batch_size, sampler=test_sampler,
        num_workers=config.num_workers, pin_memory=True, collate_fn=utils.collate_fn
    )
    
    model = get_object_detection_model(input_size=config.input_size, num_classes=config.num_classes)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    scaler = torch.cuda.amp.GradScaler()
    
    start_epoch = load_checkpoint(model, optimizer, config.load_checkpoint_path) if config.load_checkpoint_path else 0
    
    for epoch in range(start_epoch, config.num_epochs):
        train_sampler.set_epoch(epoch)
        
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=config.print_freq, scaler=scaler)
        lr_scheduler.step()
        
        evaluate(model, data_loader_test, device=device)
        
        if config.save_checkpoint_path and rank == 0:
            save_checkpoint(model, optimizer, epoch + 1, f"{config.save_checkpoint_path[:-4]}_{epoch+1}.pth")
            print(f"Checkpoint saved to {config.save_checkpoint_path[:-4]}_{epoch+1}.pth")

        if rank == 0:
            coco_eval = evaluate(model, data_loader_test, device=device)
            metrics = {
                'Epoch': epoch,
                'AP': coco_eval.stats[0],
                'AP50': coco_eval.stats[1],
                'AP75': coco_eval.stats[2],
                'APs': coco_eval.stats[3],
                'APm': coco_eval.stats[4],
                'APl': coco_eval.stats[5],
                'AR1': coco_eval.stats[6],
                'AR10': coco_eval.stats[7],
                'AR100': coco_eval.stats[8],
                'ARs': coco_eval.stats[9],
                'ARm': coco_eval.stats[10],
                'ARl': coco_eval.stats[11]
            }
            wandb.log(metrics)

        if epoch % 2 == 0 and rank == 0:
            log_sample_predictions(model, data_loader_test, device, num_images=10)

    print('=' * 100)
    print('FINAL EVAL ' * 10)
    print('=' * 100)

    if rank == 0:
        wandb.finish()
    print("Training completed!")

if __name__ == "__main__":
    main()