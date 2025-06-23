#!/bin/bash

# make sure uv is installed, if not:
# curl -LsSf https://astral.sh/uv/install.sh | sh

# for MacOS users:
# brew install libomp

set -e

MINIMAL=false
W_FLASH_ATTN=false
VENV_PATH="./erwin"
PYTHON_BIN_PATH=$(which python)


# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --minimal)
            MINIMAL=true
            shift
            ;;
        --with-flash-attn)
            W_FLASH_ATTN=true
            shift
            ;;
        --venv)
            VENV_PATH="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--venv VENV_PATH] [--minimal]"
            exit 1
            ;;
    esac
done


# create and check in the target python virtual environment
uv venv "${VENV_PATH}" --python "${PYTHON_BIN_PATH}"
source "${VENV_PATH}/bin/activate"


if [ "$MINIMAL" = true ]; then
    echo "Installing minimal dependencies [Erwin]"
    uv pip install torch==2.5.0
    uv pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu124.html # CUDA 12.4
    uv pip install numpy
    uv pip install einops

    if [ "$W_FLASH_ATTN" = true ]; then
        uv pip install flash-attn
    fi

    uv pip install balltree-erwin
else
    echo "Installing all dependencies [Erwin + baselines + experiments]"

    # Erwin dependencies
    uv pip install torch==2.5.0
    uv pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu124.html # CUDA 12.4
    uv pip install numpy
    uv pip install einops

    if [ "$W_FLASH_ATTN" = true ]; then
        uv pip install flash-attn
    fi

    uv pip install balltree-erwin

    # PointTransformer v3 dependencies
    uv pip install addict
    uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html # CUDA 12.4
    uv pip install spconv-cu120
    uv pip install timm

    # MD dependencies
    uv pip install h5py
    # cosmology dependencies
    uv pip install tensorflow
    # ShapeNet-Car dependencies
    # see https://github.com/ml-jku/UPT/blob/main/SETUP_DATA.md

    # misc dependencies
    uv pip install wandb
    uv pip install tqdm
    uv pip install matplotlib
fi
