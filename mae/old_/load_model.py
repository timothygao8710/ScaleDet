import timm
import torch

import models_vit

input_size = 224

model = models_vit.__dict__["vit_large_patch16"](
    img_size=input_size,
    num_classes=60,
    global_pool=False,
)

# Path to your pretrained weights file
pretrained_weights_path = '/home/timothygao/scalemae_docker/weights/scalemae-vitlarge-800.pth'

checkpoint = torch.load(pretrained_weights_path, map_location="cpu")
checkpoint_model = checkpoint["model"]
state_dict = model.state_dict()
for k in ["head.weight", "head.bias"]:
    if (
        k in checkpoint_model
        and checkpoint_model[k].shape != state_dict[k].shape
    ):
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

if input_size != 224:
    print("WRONG IMAGE SIZE", "CURRENT positional embeddings onkly work with 224")
    if (
        "pos_embed" in checkpoint_model
        and checkpoint_model["pos_embed"].shape != state_dict["pos_embed"].shape
    ):
        print(f"Removing key pos_embed from pretrained checkpoint")
        del checkpoint_model["pos_embed"]


# Load the pretrained model
msg = model.load_state_dict(checkpoint_model, strict=False)
print(msg)

# print(models_vit.__dict__.keys())