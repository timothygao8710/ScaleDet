#!/bin/bash

echo "staring"
# --no-cache \
docker build \
    --progress=plain \
    --network host \
    --platform linux/amd64 \
    -t=scalemae .
echo "Done Bulding"

