# Distributed Inference with PyTorch and LiteLLM

This project demonstrates a distributed inference setup using PyTorch and LiteLLM, allowing for efficient processing across multiple GPU nodes.

## Project Structure

```
distributed-inference-pytorch-litellm/
│
├── src/
│   ├── distributed_inference.py
│   ├── utils.py
│   └── config.py
│
├── scripts/
│   ├── run_node0.sh
│   └── run_node1.sh
│
├── tests/
│   └── test_distributed_inference.py
│
├── docs/
│   ├── setup_guide.md
│   └── troubleshooting.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Key Files

### README.md

```markdown
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

### src/distributed_inference.py

```python
import os
import sys
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import litellm
from litellm import completion
import logging
import datetime
from utils import setup_logging, gpu_tensor_operation
from config import CONFIG

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = CONFIG['MASTER_ADDR']
    os.environ['MASTER_PORT'] = CONFIG['MASTER_PORT']
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.barrier()

def cleanup():
    dist.destroy_process_group()

class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}

def get_model_response(prompt):
    messages = [{"content": prompt, "role": "user"}]
    try:
        response = completion(CONFIG['MODEL_NAME'], messages)
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error getting model response: {e}")
        return "Error: Unable to get model response"

def main():
    setup_logging()
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        setup(rank, world_size)

        device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        os.environ["OPENAI_API_KEY"] = CONFIG['API_KEY']
        litellm.api_base = CONFIG['API_BASE']

        dataset = load_dataset("imdb", split="train[:1%]")
        custom_dataset = CustomDataset(dataset["text"], dataset["label"])
        train_sampler = torch.utils.data.distributed.DistributedSampler(custom_dataset, num_replicas=world_size, rank=rank)
        train_dataloader = DataLoader(custom_dataset, batch_size=4, sampler=train_sampler)

        for epoch in range(3):
            logging.info(f"Starting epoch {epoch}")
            train_sampler.set_epoch(epoch)
            for batch in train_dataloader:
                prompts = batch["text"]
                labels = batch["label"]
                
                gpu_results = [gpu_tensor_operation(prompt, device) for prompt in prompts]
                responses = [get_model_response(prompt) for prompt in prompts]
                
                if rank == 0:
                    for prompt, response, label, gpu_result in zip(prompts, responses, labels, gpu_results):
                        logging.info(f"Prompt: {prompt[:100]}...")
                        logging.info(f"Response: {response[:100]}...")
                        logging.info(f"Label: {label}")
                        logging.info(f"GPU Result: {gpu_result}\n")

        cleanup()
    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### requirements.txt

```
torch>=1.8.0
litellm
datasets
```

### .gitignore

```
# Python
__pycache__/
*.py[cod]
*.so

# Virtual Environment
venv/
env/
*.env

# IDEs
.vscode/
.idea/

# Logs
*.log

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
config.py
```
