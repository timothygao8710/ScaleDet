#!/bin/bash

echo ""
echo "Check build info (make sure CUDA version is correct, etc..."
python -c "import torch; print(torch.__config__.show())"

echo ""
echo "Check if cuda is available..."
python -c "import torch; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Using device:', device); torch.rand(10).to(device)"

echo ""
echo "Check CUDA version..."
cat /usr/local/cuda/version.txt

echo ""
nvidia-smi

bash ./scripts/run_knn_eval.sh
