#!/bin/bash

export NCCL_SOCKET_IFNAME=ens3
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

# Replace with your actual API key
export OPENAI_API_KEY="your_api_key_here"

torchrun \
  --nproc_per_node=1 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=149.165.173.22 \
  --master_port=29500 \
  src/distributed_finetuning.py