FROM --platform=linux/amd64 nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 AS build-image

# These vars are available during docker build, but when running the container:
ARG PYTHON_VERSION=3.10
ARG CONDA_PYTHON_VERSION=3
ARG CONDA_ENV_NAME=scalemae
ARG CONDA_DIR="/opt/miniconda-latest"
ARG CONDA_INSTALLER="/tmp/miniconda.sh"
ARG PATH="${CONDA_DIR}/bin:$PATH"
ARG DEBIAN_FRONTEND=noninteractive

# These vars are available from within the container when its running (after build):
ENV FORCE_CUDA="1"
ENV TERM=xterm
ENV PATH="${CONDA_DIR}/bin:$PATH"

# Default docker build shell is sh, change it to bash:
SHELL ["/bin/bash", "-c"]

RUN echo "Shell: $SHELL"
RUN echo "Python version: ${PYTHON_VERSION}"

# Optional: Not dependencies for scalemae, but useful to have in deployment:
COPY ./install_sys_utils.sh .
RUN /bin/bash ./install_sys_utils.sh

# This link Explains the DEBIAN_FRONTEND tzdata line: https://stackoverflow.com/a/44333806
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata && \
    apt install build-essential software-properties-common -y && \
    apt install -y -q --no-install-recommends \
        bzip2 \
        ca-certificates \
        curl \
        tzdata \
        ffmpeg \
        libsm6 \
        libxext6 \
        libglib2.0-0 \
        libxrender1 && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies.
RUN echo "Downloading Miniconda installer ..."
RUN curl -fsSL -o "${CONDA_INSTALLER}" "https://repo.anaconda.com/miniconda/Miniconda${CONDA_PYTHON_VERSION}-latest-Linux-x86_64.sh"
RUN bash "${CONDA_INSTALLER}" -b -p "${CONDA_DIR}" && \
    rm -f "${CONDA_INSTALLER}" && \
    rm -rf /tmp/*

RUN conda update -yq -n base conda && \
    conda init bash && \
    source "${CONDA_DIR}/etc/profile.d/conda.sh" && \
    source "${CONDA_DIR}/etc/profile.d/conda.sh"

RUN conda config --system --prepend channels conda-forge && \
    conda config --set channel_priority strict && \
    conda config --system --set auto_update_conda false && \
    conda config --system --set show_channel_urls true && \
    conda install -n base conda-libmamba-solver && \
    conda config --set solver libmamba

RUN conda create -n "${CONDA_ENV_NAME}" python=3.10 && \
    conda install -y --name "${CONDA_ENV_NAME}" "geopandas" && \
    conda install -y --name "${CONDA_ENV_NAME}" \
    pytorch==1.13.1=py3.10_cuda11.7_cudnn8.5.0_0 \
    torchvision==0.14.1 \
    pytorch-cuda=11.7 \
    -c pytorch -c nvidia && \
    conda run -n "${CONDA_ENV_NAME}" python -m pip install --no-cache-dir  \
        "numpy>=1.21.0,<2.0.0" \
        "timm==0.6.12" \
        "wandb" \
        "pre-commit>=2.15.0,<3.0.0" \
        "black[jupyter]>=21.8,<23" \
        "flake8>=3.8,<5" \
        "isort[colors]>=5.8,<6" \
        "pydocstyle[toml]>=6.1,<7" \
        "pyupgrade>=1.24,<3" \
        "pytest>=6.1.2,<8" \
        "pytest-cov>=2.4,<4" \
        "types-requests>=2.28.9,<2.30.0" \
        "types-python-dateutil>=2.8.19,<2.9.0" \
        "kornia==0.6.9" \
        "torchgeo==0.5.2" \
        "classy-vision==0.6.0" \
        "tensorboard" \
        "opencv-python"

# Clean up
RUN sync && \
    conda clean --all --yes && \
    sync && \
    rm -rf ~/.cache/pip/*

WORKDIR /proj/scalemae
COPY ./ ./
COPY ./dotfiles/* /root/

# Source bashrc, which has the conda activate and init stuff. Allows us to use `conda activate` on subsequent lines:
SHELL ["/bin/bash", "-c", "source /root/.bashrc"]

RUN conda activate "${CONDA_ENV_NAME}"
RUN pip install --upgrade pip && \
    pip install wheel setuptools ninja && \
    pip install -e .

CMD ["bash"]

