import os
import json
import torch
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

class COCOFormatDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_file, transforms=None):
        self.transforms = transforms
        self.picture_dir = image_dir
        self.json_path = label_file
        
        with open(self.json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        self.image_to_anns = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_to_anns:
                self.image_to_anns[image_id] = []
            self.image_to_anns[image_id].append(ann)
        
        self.image_ids = [img['id'] for img in self.coco_data['images'] if img['id'] in self.image_to_anns]
        self.id_to_filename = {img['id']: img['file_name'] for img in self.coco_data['images'] if img['id'] in self.image_to_anns}

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_filename = self.id_to_filename[image_id]
        img_path = os.path.join(self.picture_dir, img_filename)
        
        img = read_image(img_path)
        
        anns = self.image_to_anns[image_id]
        
        num_objs = len(anns)
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            boxes.append(ann['bbox'])
            boxes[-1][2] += boxes[-1][0]
            boxes[-1][3] += boxes[-1][1]
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann["iscrowd"])
        
        boxes = torch.as_tensor(boxes)
        labels = torch.as_tensor(labels)
        areas = torch.as_tensor(areas)
        iscrowd = torch.as_tensor(iscrowd)
        image_id = torch.tensor([image_id])
        
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