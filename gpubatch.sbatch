#!/bin/bash

# This script will request one GPU device and 1 CPU core

#SBATCH --job-name=gpuSbatch
#SBATCH --output=/project/rkessler/djbrout/deepsnid/batchout/tanh_10k.out
#SBATCH --error=/project/rkessler/djbrout/deepsnid/batchout/tanh_10k.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load python

# if your executable was built with CUDA, be sure to load the CUDA module:
module load cuda

# if your exectuable was built with PGI (OpenACC), be sure to load the PGI module:
# module load pgi/2013

#
# your GPU-based executable here
#
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mlp2.py
