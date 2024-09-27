import unittest
import torch
import torch.distributed as dist
from src.distributed_finetuning import setup, cleanup, CustomDataset
from src.utils import process_batch, gpu_tensor_operation

class TestDistributedFinetuning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This setup assumes running on a single machine with multiple processes
        cls.world_size = 2
        cls.rank = 0
        setup(cls.rank, cls.world_size)

    @classmethod
    def tearDownClass(cls):
        cleanup()

    def test_custom_dataset(self):
        texts = ["text1", "text2", "text3"]
        labels = [0, 1, 0]
        dataset = CustomDataset(texts, labels)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[1]["text"], "text2")
        self.assertEqual(dataset[1]["label"], 1)

    def test_process_batch(self):
        def mock_get_model_response(prompt):
            return "mocked response"

        model = torch.nn.Linear(1, 1)  # Dummy model
        prompts = ["prompt1", "prompt2"]
        labels = [0, 1]
        loss = process_batch(model, prompts, labels, mock_get_model_response)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_gpu_tensor_operation(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            result = gpu_tensor_operation("test", device)
            self.assertIsInstance(result, float)
        else:
            print("CUDA not available, skipping GPU test")

if __name__ == '__main__':
    unittest.main()