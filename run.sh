#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export WANDB_MODE=offline
export OMP_NUM_THREADS=1

nvidia-smi
source bashrc_path

python3 ./audioldm_train/train/latent_diffusion.py -c ./audioldm_train/config/mos_as_token/qa_mdt.yaml