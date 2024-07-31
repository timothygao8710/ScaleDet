import torch
import torch.nn as nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from models_vit import vit_large_patch16
from torchvision.models.detection import FasterRCNN
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict
from torch.profiler import profile, record_function, ProfilerActivity

class ScaleMAEBackbone(nn.Module):
    def __init__(self, num_classes, input_size, pretrained_weights_path=None):
        super(ScaleMAEBackbone, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.backbone = vit_large_patch16(
            img_size=input_size,
            num_classes=0,
            global_pool=False,
        )
        
        if pretrained_weights_path:
            self.load_pretrained_weights(pretrained_weights_path)
                
        self.embed_dim = 1024
        self.image_res = None

    def load_pretrained_weights(self, pretrained_weights_path):
        state_dict = torch.load(pretrained_weights_path, map_location="cpu")
        model_state_dict = self.backbone.state_dict()
        
        for k in ["head.weight", "head.bias"]:
            if k in state_dict and model_state_dict[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del state_dict[k]

        if self.input_size != 224:
            print("WARNING: Current positional embeddings only work with 224x224 input size")
            if "pos_embed" in state_dict and state_dict["pos_embed"].shape != model_state_dict["pos_embed"].shape:
                print(f"Removing key pos_embed from pretrained checkpoint")
                del state_dict["pos_embed"]
                
        msg = self.backbone.load_state_dict(state_dict, strict=False)
        del state_dict
        
    def forward(self, x):
        input_res = torch.tensor([self.img_res], device=x.device)
        x = self.backbone.forward_features(x, input_res)        
        return x

# https://arxiv.org/pdf/2203.16527
class SimpleFPN(nn.Module):
    def __init__(self, backbone, out_channels=256):
        super(SimpleFPN, self).__init__()
        self.backbone = backbone
        self.out_channels = out_channels
        self.conv_1_2 = nn.Conv2d(1024, out_channels, kernel_size=1, stride=2)
        self.conv_1 = nn.Conv2d(1024, out_channels, kernel_size=1, stride=1)
        self.deconv_2 = nn.ConvTranspose2d(1024, out_channels, kernel_size=2, stride=2)
        self.deconv_4 = nn.ConvTranspose2d(1024, out_channels, kernel_size=4, stride=4)
        
    def forward(self, x):
        scalemae_result = self.backbone(x)
        # print(f"Shape of ScaleMAE output {scalemae_result.shape}")
        batch, num_patches, embed_dim = scalemae_result.shape
        patch_len = int(num_patches ** 0.5)
        scalemae_result = scalemae_result.view(batch, patch_len, patch_len, embed_dim)
        scalemae_result = scalemae_result.permute(0, 3, 1, 2).contiguous()
        # print(f"Shape of reshaped ScaleMAE output {scalemae_result.shape}")
        feature_map = OrderedDict()
        feature_map['0'] = self.conv_1_2(scalemae_result)
        feature_map['1'] = self.conv_1(scalemae_result)
        feature_map['2'] = self.deconv_2(scalemae_result)
        feature_map['3'] = self.deconv_4(scalemae_result)
        return feature_map

class ViTDet(nn.Module):
    def __init__(self, input_size, num_classes, pretrained_weights_path=None):
        super(ViTDet, self).__init__()
        
        self.backbone = ScaleMAEBackbone(
            num_classes=num_classes,
            input_size=input_size,
            pretrained_weights_path=pretrained_weights_path
        )
        
        # https://github.com/ViTAE-Transformer/ViTDet/blob/main/configs/ViTDet/ViTDet-ViT-Base-100e.py
        self.fpn_adaptor = SimpleFPN(self.backbone, 256)
                
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128),(64, 128, 256),(128, 256, 512),(128, 256, 512)),
            aspect_ratios=((0.5, 1.0, 2.0),(0.5, 1.0, 2.0),(0.5, 1.0, 2.0),(0.5, 1.0, 2.0),),
        )

        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=0
        )
        
        self.rcnn = FasterRCNN(
            backbone=self.fpn_adaptor,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )

    def forward(self, images, targets=None, image_res=0.3):
        self.backbone.img_res = image_res
        return self.rcnn(images, targets)

def get_object_detection_model(input_size, num_classes):
    if input_size != 224:
        print("Warning: Model works best with input size 224, will need to modify canonical_scale in MultiScaleRoIAllign")
    pretrained_weights_path = '/home/timothygao/scalemae_docker/weights/scalemae-unwrapped.pth'
    model = ViTDet(num_classes=num_classes, input_size=input_size, pretrained_weights_path=pretrained_weights_path)
    return model

def print_memory_stats(step):
    print(f"\n--- {step} ---")
    print(f"Current memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    print(f"Current memory reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")
    print(f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1e6:.2f} MB")

if __name__ == '__main__':
    num_classes = 60
    input_size = 224
    pretrained_weights_path = '/home/timothygao/scalemae_docker/weights/scalemae-unwrapped.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.cuda.reset_peak_memory_stats()  # Reset peak stats
    print_memory_stats("Before model initialization")

    model = ViTDet(num_classes=num_classes, input_size=input_size, pretrained_weights_path=pretrained_weights_path).to(device)
    print_memory_stats("After model initialization")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print_memory_stats("After optimizer creation")

    dummy_input = torch.randn(2, 3, input_size, input_size).to(device)
    dummy_targets = [
        {
            "boxes": torch.tensor([[50, 50, 100, 100], [30, 30, 70, 70]], dtype=torch.float32).to(device),
            "labels": torch.tensor([1, 2], dtype=torch.int64).to(device),
            "masks": torch.randint(0, 2, (2, input_size, input_size), dtype=torch.uint8).to(device)
        },
        {
            "boxes": torch.tensor([[60, 60, 120, 120], [40, 40, 80, 80]], dtype=torch.float32).to(device),
            "labels": torch.tensor([1, 2], dtype=torch.int64).to(device),
            "masks": torch.randint(0, 2, (2, input_size, input_size), dtype=torch.uint8).to(device)
        }
    ]
    print_memory_stats("After creating dummy inputs")

    model.train()
    print_memory_stats("Before model inference")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference_and_backward"):
            optimizer.zero_grad()
            output = model(dummy_input, dummy_targets)
            print_memory_stats("After forward pass")

            loss = sum(loss for loss in output.values())
            print_memory_stats("After loss computation")

            loss.backward()
            print_memory_stats("After backward pass")

            optimizer.step()
            print_memory_stats("After optimizer step")

    print_memory_stats("After model inference and backward")

    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))