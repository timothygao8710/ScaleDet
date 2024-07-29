#!/bin/bash

# Make script fail on first error:
set -e

apt update
apt install --no-install-recommends -y software-properties-common
add-apt-repository ppa:git-core/ppa

apt update -y
apt full-upgrade -y
apt install --no-install-recommends -y \
    bash-completion \
    build-essential \
    bzip2 \
    curl \
    diffutils \
    git \
    gzip \
    htop \
    jpegoptim \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    make \
    nano \
    optipng \
    pigz \
    rsync \
    silversearcher-ag \
    sudo \
    tmux \
    tmuxp \
    tree \
    unzip \
    vim \
    wget \
    xclip \
    zip

apt-get clean
rm -rf /var/lib/apt/lists/*
