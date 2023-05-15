import os
from datetime import datetime
import torch


class Config:
    def __init__(
        self,
        cpu: bool = False,
        output_path: str = "./output",
        train_mode=True,
        lr=5e-3,
        epoch=1000,
        batch_size=1000,
        valid_split=0.2,
    ):
        # Set the device to use for training
        self.device = torch.device(
            "cpu" if (cpu or not torch.cuda.is_available()) else "cuda"
        )
        # Set hyperparameters
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.valid_split = valid_split
        # Set output directory path and create the directory if it doesn't exist
        if train_mode:
            now = datetime.now()
            self.output_path = os.path.join(output_path, now.strftime("%Y%m%d_%H%M"))
            os.makedirs(self.output_path, exist_ok=True)
