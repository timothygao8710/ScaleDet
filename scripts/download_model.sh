#!/bin/bash

# Get the directory of this script so that we can reference paths correctly no matter which folder
# the script was launched from:
SCRIPT_DIR="$(builtin cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "SCRIPT_DIR: ${SCRIPT_DIR}"
PROJ_ROOT=$(realpath ${SCRIPT_DIR}/..)
echo "PROJ_ROOT: ${PROJ_ROOT}"

pushd "${PROJ_ROOT}/data"
URL="https://github.com/bair-climate-initiative/scale-mae/releases/download/base-800/scalemae-vitlarge-800.pth"
wget "${URL}"
