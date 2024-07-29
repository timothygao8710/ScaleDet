import os
import json
import torch
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

class COCOFormatDataset(torch.utils.data.Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        self.root_dir = '/home/timothygao/xview_400_0'
        self.picture_dir = '/home/timothygao/xview_400_0/train_images_400_0'
        self.json_path = '/home/timothygao/xview_400_0/train_400_0.json'
        
        # Load COCO format annotations
        with open(self.json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create a mapping from image_id to annotations
        self.image_to_anns = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_to_anns:
                self.image_to_anns[image_id] = []
            self.image_to_anns[image_id].append(ann)
        
        # Filter out images with no annotations
        self.image_ids = [img['id'] for img in self.coco_data['images'] if img['id'] in self.image_to_anns]
        self.id_to_filename = {img['id']: img['file_name'] for img in self.coco_data['images'] if img['id'] in self.image_to_anns}

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_filename = self.id_to_filename[image_id]
        img_path = os.path.join(self.picture_dir, img_filename)
        
        # Load image
        img = read_image(img_path)
        
        # Get annotations for this image
        anns = self.image_to_anns[image_id]
        
        # Prepare target
        num_objs = len(anns)
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            boxes.append(ann['bbox'])
            # COCO format uses [x, y, width, height], convert to [x1, y1, x2, y2]
            boxes[-1][2] += boxes[-1][0]
            boxes[-1][3] += boxes[-1][1]
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann["iscrowd"])
        
        # Convert to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.int16)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float16)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.bool)
        image_id = torch.tensor([image_id])
        
        # Wrap image and targets into torchvision tv_tensors
        img = tv_tensors.Image(img)
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.image_ids)

if __name__ == '__main__':
    import utils
    dataset = COCOFormatDataset()
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=utils.collate_fn
    )
    # For Training
    images, targets = next(iter(data_loader))