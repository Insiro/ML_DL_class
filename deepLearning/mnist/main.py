from typing import Any
from datetime import datetime
import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torchmetrics import Accuracy
import json


class Config:
    def __init__(
        self, cpu: bool = False, output_path: str = "./output", train_mode=True
    ):
        # Set the device to use for training
        self.device = torch.device(
            "cpu" if (cpu or not torch.cuda.is_available()) else "cuda"
        )
        # Set hyperparameters
        self.lr = 5e-3
        self.epoch = 1000
        self.batch_size = 1000
        self.valid_split = 0.2
        # Set output directory path and create the directory if it doesn't exist
        if train_mode:
            now = datetime.now()
            self.output_path = os.path.join(output_path, now.strftime("%Y%m%d_%H%M"))
            os.makedirs(self.output_path, exist_ok=True)


class LeNetLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # Define the convolutional layer, batch normalization layer, ReLU activation function, and max pooling layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class LeNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Define the layers of the LeNet network
        self.layer1 = LeNetLayer(1, 6)
        self.layer2 = LeNetLayer(6, 16)
        self.layer3 = LeNetLayer(16, 120)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(1080, 84)
        self.fc_2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x


class SaveModelCallback:
    def __init__(self, config: Config, minimum_epoch: int = 100) -> None:
        # Initialize variables for saving the best model and results
        self.best_loss: Any = np.inf
        self.save_path = config.output_path
        self.minimum_epoch = minimum_epoch
        self.acc: Any = 0
        self.best_epoch: Any = np.inf

    def on_step_end(self, model, epoch: int, loss, acc) -> None:
        # Save the model at the end of every epoch, and the best model based on the loss function
        if epoch < self.minimum_epoch:
            return
        # determine save file name
        output_name = f"{epoch}_{loss}_{acc}.pth"
        torch.save(model.state_dict(), os.path.join(self.save_path, output_name))
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            torch.save(
                model.state_dict(),
                os.path.join(self.save_path, f"best.pth"),
            )

    def on_train_end(self) -> None:
        # save the training result
        with open(os.path.join(self.save_path, "result.json"), "w") as f:
            json.dump(
                {
                    "acc": self.acc.item(),
                    "loss": self.best_loss.item(),
                    "epoch": self.best_epoch,
                },
                f,
            )


class TensorBoardWriter:
    def __init__(self) -> None:
        # Initialize the TensorBoard writer
        self.tb_writer = SummaryWriter()

    def log_step(self, train_acc, train_loss, test_acc, test_loss, lr, step):
        # Log the training and test results
        self.tb_writer.add_scalar(f"Loss/train_loss", train_loss, step)
        self.tb_writer.add_scalar(f"Loss/test_loss", test_loss, step)
        self.tb_writer.add_scalar(f"Acc/train_ccc", train_acc, step)
        self.tb_writer.add_scalar(f"Acc/test_acc", test_acc, step)
        self.tb_writer.add_scalar(f"lr", lr, step)
        # Print the results
        print(f"train acc: {train_acc} loss: {train_loss}, lr : {lr}")
        print(f"test acc: {test_acc} loss: {test_loss}")

    def flush(self):
        self.tb_writer.flush()


class Trainer:
    def __init__(
        self, model: nn.Module, config: Config, loss_fn, acc_fn, scheduler
    ) -> None:
        # Initialize the model, loss function, and optimizer
        self.model: nn.Module = model
        self.config = config
        self.logger = TensorBoardWriter()
        self.loss_fn = loss_fn
        self.acc_fn = acc_fn
        self.scheduler = scheduler
        self.callbacks: list[Any] = []

    def add_callback(self, callback) -> None:
        self.callbacks.append(callback)

    def __calc_matrix(self, pred, y) -> tuple:
        # Calculate the loss and accuracy
        cost = self.loss_fn(pred, y)
        acc = self.acc_fn(pred, y)
        return acc, cost

    def __train_step(self, dataset, optim, step: int) -> tuple:
        # Train the model for one epoch
        self.model.train()
        # Iterate over the dataset
        for x, y in tqdm(dataset, desc=f"train step {step}/{self.config.epoch}"):
            # Move the data to the device
            x = x.to(self.config.device)
            y = y.to(self.config.device)
            # Clear the gradients
            optim.zero_grad()
            pred = self.model(x)
            acc, cost = self.__calc_matrix(pred, y)
            # Backpropagate the loss
            cost.backward()
            # Update the parameters
            optim.step()
        return acc, cost

    def __test_step(self, dataset, step: int, max_step: int, model=None) -> tuple:
        # Test the model for one epoch
        if model is None:
            model = self.model
        model.eval()
        for x, y in tqdm(dataset, desc=f"test step {step}/{max_step}"):
            # Move the data to the device
            x = x.to(self.config.device)
            y = y.to(self.config.device)
            # Calculate the loss and accuracy
            pred = self.model(x)
            acc, cost = self.__calc_matrix(pred, y)
        return acc, cost

    def train(self, train_loader, val_loader, optim) -> None:
        # Train the model for epochs
        for step in range(self.config.epoch):
            lr = self.scheduler.get_last_lr()[0]
            # train step
            train_acc, train_loss = self.__train_step(train_loader, optim, step)
            self.scheduler.step()
            # test step
            test_acc, test_loss = self.__test_step(val_loader, step, self.config.epoch)
            self.logger.log_step(train_acc, train_loss, test_acc, test_loss, lr, step)
            # callback on step end
            for callback in self.callbacks:
                callback.on_step_end(self.model, step, test_loss, test_acc)
        # callback on train end
        for callback in self.callbacks:
            callback.on_train_end()
        # flush the logger
        self.logger.flush()

    def test(self, test_loader, model):
        # Test the model
        print("validate mode")
        test_acc, test_loss = self.__test_step(test_loader, 1, 1, model)
        print(f"test acc: {test_acc} loss: {test_loss}")


def main():
    config = Config()
    # region data loader
    train_set = datasets.MNIST(
        "./data", train=True, download=True, transform=transforms.ToTensor()
    )
    test_set = datasets.MNIST(
        "./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    # split the train set into train and validation set
    data_size = len(train_set)
    valid_size = int(data_size * config.valid_split)
    train_set, val_set = random_split(train_set, [data_size - valid_size, valid_size])
    # create data loaders
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(val_set, batch_size=config.batch_size)
    test_loader = DataLoader(test_set, batch_size=config.batch_size)
    # endregion
    num_classes = len(train_set.dataset.classes)
    # create the model, optimizer, and trainer instance
    model = LeNet(num_classes)
    model.to(config.device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=config.lr)

    trainer = Trainer(
        model=model,
        config=config,
        loss_fn=nn.CrossEntropyLoss(),
        acc_fn=Accuracy(task="multiclass", num_classes=num_classes).to(config.device),
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8),
    )
    trainer.add_callback(SaveModelCallback(config))
    # train and test the model
    trainer.train(train_loader, valid_loader, optimizer)
    model.load_state_dict(torch.load(os.path.join(config.model_path, "best.pt")))
    trainer.test(test_loader, model)


if __name__ == "__main__":
    main()
