"""
Nanotron training script.

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=8 run_train.py --config-file examples/config_tiny_llama.yaml
```
"""
import argparse
from nanotron import logging
from nanotron.trainer import DistributedTrainer
from nanotron.dataloader import get_train_dataloader, get_valid_dataloader

logger = logging.get_logger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # Load trainer and data
    trainer = DistributedTrainer(config_file)
    train_dataloader = get_train_dataloader(trainer)
    valid_dataloader = get_valid_dataloader(trainer)

    # Train
    trainer.train(train_dataloader, valid_dataloader)
