
#!/bin/bash

# Get the directory of this script so that we can reference paths correctly no matter which folder
# the script was launched from:
SCRIPT_DIR="$(builtin cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT=$(realpath ${SCRIPT_DIR}/..)
WEIGHTS_DIR="${PROJ_ROOT}/weights"
DATA_DIR="${PROJ_ROOT}/data"
echo "SCRIPT_DIR: ${SCRIPT_DIR}"
echo "PROJ_ROOT: ${PROJ_ROOT}"
echo "WEIGHTS_DIR: ${WEIGHTS_DIR}"
echo "DATA_DIR: ${DATA_DIR}"

pushd "${PROJ_ROOT}/mae"
# python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 \
python \
    main_pretrain.py \
    --config "config/fmow.yaml" \
    --eval_dataset "resisc"  \
    --eval_train_fnames ./splits/train-resisc.txt  \
    --eval_val_fnames ./splits/val-resisc.txt

df -P -T /datasets | tail -n +2 | awk '{print $2}'
