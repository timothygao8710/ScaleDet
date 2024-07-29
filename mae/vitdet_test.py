import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.anchor_utils import AnchorGenerator
import models_vit
from torchvision.models.detection import MaskRCNN, FasterRCNN
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict
from torch.profiler import profile, record_function, ProfilerActivity

# ScaleMAE Det: Make Your ViT Detector Scale Invarient
class ScaleMAEBackbone(nn.Module):
    def __init__(self, num_classes, input_size, pretrained_weights_path=None):
        super(ScaleMAEBackbone, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.backbone = models_vit.__dict__["vit_large_patch16"](
            img_size=input_size,
            num_classes=0,
            global_pool=False,
        )

        # Load pretrained weights if provided
        if pretrained_weights_path:
            self.load_pretrained_weights(pretrained_weights_path)


        self.embed_dim = self.backbone.embed_dim
        self.img_res = 0.3

    def load_pretrained_weights(self, pretrained_weights_path):
        checkpoint = torch.load(pretrained_weights_path, map_location="cpu")
        checkpoint_model = checkpoint["model"]
        state_dict = self.backbone.state_dict()

        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        if self.input_size != 224:
            print("WARNING: Current positional embeddings only work with 224x224 input size")
            if "pos_embed" in checkpoint_model and checkpoint_model["pos_embed"].shape != state_dict["pos_embed"].shape:
                print(f"Removing key pos_embed from pretrained checkpoint")
                del checkpoint_model["pos_embed"]

        msg = self.backbone.load_state_dict(checkpoint_model, strict=False)
        # print(msg)

    def forward(self, x):
        #### CHANGE THIS ####
        input_res = torch.tensor([self.img_res], device=x.device)
        #####################

        x = self.backbone.forward_features(x, input_res)

        # print(f"ScaleMAE Output Shape {x.shape}")
        
        return x

class FPNAdaptor(nn.Module):
    def __init__(self, backbone, out_channels):
        super(FPNAdaptor, self).__init__()

        self.out_channels = out_channels
        self.backbone = backbone

        # TODO: Try cloning inputs, also, extract from intermediate layers of transformer
        # in_channels_list = [backbone.embed_dim]
        
        in_channels_list = [2048, 1024, 512, 256]
        # in_channels_list = [1024]

        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels
        )
        
        self.conv_1_2 = nn.Conv2d(1024, 2048, kernel_size=1, stride=2)
        self.conv_1 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.deconv_2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.deconv_4 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=4)

    def forward(self, x):
        scalemae_result = self.backbone(x)

        # Exclude CLS token
        scalemae_result = scalemae_result[:, 1:, :]

        # print(f"Shape of ScaleMAE output {scalemae_result.shape}")
        batch, num_patches, embed_dim = scalemae_result.shape
        temp = int(num_patches ** 0.5)

        scalemae_result = scalemae_result.view(batch, temp, temp, embed_dim)
        # scalemae_result = torch.einsum('b i j e -> b e i j', scalemae_result)
        scalemae_result = scalemae_result.permute(0, 3, 1, 2).contiguous()

        # print(f"Shape of reshaped ScaleMAE output {scalemae_result.shape}")

        # TODO: Try 4 of these
        feature_map = OrderedDict()
        # feature_map['0'] = scalemae_result
        
        feature_map['0'] = self.conv_1_2(scalemae_result).contiguous()
        feature_map['1'] = self.conv_1(scalemae_result).contiguous()
        feature_map['2'] = self.deconv_2(scalemae_result).contiguous()
        feature_map['3'] = self.deconv_4(scalemae_result).contiguous()

        res = self.fpn(feature_map)

        return res

class ViTDet(nn.Module):
    def __init__(self, input_size, num_classes, pretrained_weights_path=None):
        super(ViTDet, self).__init__()
        
        # Trying a image_resolution passed in as class parameter approach, may need to change later if class parameters are same across GPUs
        self.backbone = ScaleMAEBackbone(
            num_classes=num_classes,
            input_size=input_size,
            pretrained_weights_path=pretrained_weights_path
        )
        
        # Number of out channels taken from https://github.com/ViTAE-Transformer/ViTDet/blob/main/configs/ViTDet/ViTDet-ViT-Base-100e.py
        self.fpn_adaptor = FPNAdaptor(self.backbone, 256)
        
        # Anchor generator TODO: Modify according to expected object size
        # anchor_generator = AnchorGenerator(
        #     sizes=((32, 64, 128, 256),),  # Reduced sizes for smaller objects
        #     aspect_ratios=((0.5, 1, 2),)
        # )
        
        anchor_generator = AnchorGenerator(
            sizes=((64, 128, 256),(32, 64, 128),(16, 32, 64),(8, 16, 32)),  # Reduced sizes for smaller objects
            aspect_ratios=((0.5, 1, 2), (0.5, 1, 2), (0.5, 1, 2), (0.5, 1, 2))
        )
        
        # anchor_generator = AnchorGenerator(
        #     sizes=((16, 32, 64),),  # Reduced sizes for smaller objects
        #     aspect_ratios=((0.5, 1.0, 2.0),)
        # )

        # ROI Align
        roi_pooler = MultiScaleRoIAlign(
            # featmap_names=['0'],
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=0
        )
        
        # self.rcnn = MaskRCNN( MaskRCNN works better, but expects mask annotations
        self.rcnn = FasterRCNN( #TODO: Change transforms parameter to match image stand dev, mean, etc.
            backbone=self.fpn_adaptor,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            # image_mean=[0.2304, 0.1910, 0.1564],
            # image_std=[0.1710, 0.1383, 0.1277],

            # mage_mean=[0,0,0],
            # image_std=[1,1,1],
        )

    def forward(self, images, targets=None):
        self.backbone.img_res = 0.3
        
        return self.rcnn(images, targets)

def get_object_detection_model(input_size, num_classes):
    if input_size != 224:
        print("Warning: Model works best with input size 224, will need to modify canonical_scale in MultiScaleRoIAllign")

    pretrained_weights_path = '/home/timothygao/scalemae_docker/weights/scalemae-vitlarge-800.pth'

    model = ViTDet(num_classes=num_classes, input_size=input_size, pretrained_weights_path=pretrained_weights_path)
    
    return model


# import sys

# import traceback

# class TracePrints(object):
#   def __init__(self):    
#     self.stdout = sys.stdout
#   def write(self, s):
#     self.stdout.write("Writing %r\n" % s)
#     traceback.print_stack(file=self.stdout)
#   def flush(self): 
#     pass

# sys.stdout = TracePrints()



if __name__ == '__main__':
    num_classes = 91
    input_size = 800
    pretrained_weights_path = '/home/timothygao/scalemae_docker/weights/scalemae-vitlarge-800.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Before model initialization
    print(f"Memory allocated before model initialization: {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"Memory reserved before model initialization: {torch.cuda.memory_reserved() / 1e6} MB")

    model = ViTDet(num_classes=num_classes, input_size=input_size, pretrained_weights_path=pretrained_weights_path).to(device)

    # After model initialization
    print(f"Memory allocated after model initialization: {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"Memory reserved after model initialization: {torch.cuda.memory_reserved() / 1e6} MB")

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Before creating dummy inputs
    print(f"Memory allocated before creating dummy inputs: {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"Memory reserved before creating dummy inputs: {torch.cuda.memory_reserved() / 1e6} MB")

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

    # After creating dummy inputs
    print(f"Memory allocated after creating dummy inputs: {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"Memory reserved after creating dummy inputs: {torch.cuda.memory_reserved() / 1e6} MB")

    model.train()

    # Before model inference
    print(f"Memory allocated before model inference: {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"Memory reserved before model inference: {torch.cuda.memory_reserved() / 1e6} MB")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference_and_backward"):
    #         # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(dummy_input, dummy_targets)

            # Compute loss
            loss = sum(loss for loss in output.values())

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

    # After model inference and backward pass
    print(f"Memory allocated after model inference and backward: {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"Memory reserved after model inference and backward: {torch.cuda.memory_reserved() / 1e6} MB")

    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

    # # Uncomment this section if you want to check for unused parameters
    # for name, param in model.named_parameters():
    #     if param.grad is None:
    #         print(f"Unused parameter: {name}")
    #         print(f" Shape: {param.shape}")
    #         print(f" Requires grad: {param.requires_grad}")
    #         print(f" Device: {param.device}")
    #         print(f" dtype: {param.dtype}")
    #         print(f" First few values: {param.data.flatten()[:5]}")  # Show first 5 values
    #         print("---")