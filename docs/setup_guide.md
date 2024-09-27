# Setup Guide for Distributed Fine-tuning of Meta Llama 3.1 70B

This guide will walk you through the process of setting up and running the distributed fine-tuning script for the Meta Llama 3.1 70B model.

## Prerequisites

- Two or more machines with CUDA-compatible GPUs (tested on NVIDIA A100)
- Python 3.8+
- PyTorch 1.8+
- Access to the Meta Llama 3.1 70B model via a LiteLLM-compatible API

## Step 1: Environment Setup

On each node:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/distributed-finetuning-llama.git
   cd distributed-finetuning-llama
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up the necessary environment variables:
   ```
   export NCCL_SOCKET_IFNAME=ens3
   export NCCL_IB_DISABLE=1
   export NCCL_DEBUG=INFO
   export OPENAI_API_KEY="your_api_key_here"
   ```

## Step 2: Configuration

1. Update `src/config.py` with your specific settings:
   - Set the correct `MASTER_ADDR` (IP of the master node)
   - Adjust `MASTER_PORT` if needed
   - Verify the `MODEL_NAME` and `API_BASE`

2. Modify `scripts/run_node0.sh` and `scripts/run_node1.sh`:
   - Ensure the `--master_addr` matches your master node's IP
   - Adjust `--nnodes` if you're using more than two nodes

## Step 3: Running the Script

1. On the first node (master node):
   ```
   chmod +x scripts/run_node0.sh
   ./scripts/run_node0.sh
   ```

2. On the second node:
   ```
   chmod +x scripts/run_node1.sh
   ./scripts/run_node1.sh
   ```

3. If you have more nodes, create additional scripts following the pattern of `run_node1.sh`, incrementing the `--node_rank` for each.

## Step 4: Monitoring

- Check the console output on each node for progress and any error messages.
- Monitor GPU usage using `nvidia-smi`.

## Troubleshooting

If you encounter issues, refer to the `troubleshooting.md` document for common problems and their solutions.