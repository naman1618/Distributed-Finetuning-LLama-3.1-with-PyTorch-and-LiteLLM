import logging
import torch
from torch.nn.functional import cross_entropy

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def process_batch(model, prompts, labels, get_model_response):
    # This is a simplified version. In a real scenario, you'd need to handle tokenization,
    # encoding, and actual model forward pass.
    batch_loss = 0
    for prompt, label in zip(prompts, labels):
        response = get_model_response(prompt)
        # Here we're simulating a loss calculation. In reality, you'd compute this based
        # on the model's output and the true label.
        logits = torch.randn(2)  # Simulating logits for binary classification
        loss = cross_entropy(logits.unsqueeze(0), torch.tensor([label]))
        batch_loss += loss
    return batch_loss / len(prompts)

def gpu_tensor_operation(text, device):
    # This function simulates a GPU operation
    tensor = torch.tensor([ord(c) for c in text], dtype=torch.float32, device=device)
    return tensor.mean().item()