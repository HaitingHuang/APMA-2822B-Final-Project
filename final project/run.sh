#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1 --gres-flags=enforce-binding

# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 1 CPU core
#SBATCH -n 1

#SBATCH -t 00:05:00
#SBATCH -o with_gpu.out
#SBATCH -e with_gpu.err

# Load CUDA and GCC modules
module load cuda/11.7.1 
module load gcc/10.2 

# Display GPU information
nvidia-smi

# Compile CUDA part
# nvcc -c matrix_mul.cu -o matrix_mul.o
nvcc kernel.cu -o kernel.o

# Compile C++ part and link with the CUDA object file
# g++ -fopenmp main.cpp matrix_mul.o -o main.out -L/usr/local/cuda/lib64 -lcudart
# g++ -fopenmp main.cpp kernel.o -o main.out -L/usr/local/cuda/lib64 -lcudart

# Run the program
# ./main.out
# gdb ./main.out
# ./kernel.o

# Nsight
nsys profile -t cuda,nvtx ./kernel.o