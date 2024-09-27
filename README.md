# Distributed Inference with PyTorch and LiteLLM

This project demonstrates a distributed inference setup using PyTorch and LiteLLM, allowing for efficient processing across multiple GPU nodes.

## Features

- Distributed processing using PyTorch's DistributedDataParallel
- Integration with LiteLLM for API-based inference
- NCCL backend for optimal GPU communication
- Customizable for various deep learning models and tasks

## Prerequisites

- Python 3.8+
- PyTorch 1.8+
- CUDA-compatible GPUs
- Access to a LiteLLM-compatible API endpoint

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/distributed-inference-pytorch-litellm.git
   cd distributed-inference-pytorch-litellm
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Set up your environment variables:
   ```
   export NCCL_SOCKET_IFNAME=ens3
   export NCCL_IB_DISABLE=1
   export NCCL_DEBUG=INFO
   ```

2. Run the script on the first node:
   ```
   ./scripts/run_node0.sh
   ```

3. Run the script on the second node:
   ```
   ./scripts/run_node1.sh
   ```

For more detailed instructions, see the [Setup Guide](docs/setup_guide.md).

## Documentation

- [Setup Guide](docs/setup_guide.md)
- [Troubleshooting](docs/troubleshooting.md)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
Thumbs.db

# Project specific
config.py
```
