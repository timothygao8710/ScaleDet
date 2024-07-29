import torch
print(torch.cuda.is_available())  # Check if CUDA is available
print(torch.cuda.current_device())  # Check current CUDA device
print(torch.cuda.get_device_name(0))  # Get name of the CUDA device
# print(model.to('cuda'))  # Move your model to CUDA

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")