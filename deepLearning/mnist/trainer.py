from typing import Union
from tqdm import tqdm
import torch.nn as nn
from utils import Config
from callback import CallbackBase, StepState


class Trainer:
    def __init__(self, model: nn.Module, config: Config, loss_fn, acc_fn) -> None:
        # Initialize the model, loss function, and optimizer
        self.model: nn.Module = model
        self.config = config
        self.loss_fn = loss_fn
        self.acc_fn = acc_fn
        self.callbacks: list[CallbackBase] = []
        self.state = StepState()

    def add_callback(self, callback: Union[CallbackBase, list[CallbackBase]]) -> None:
        if isinstance(callback, list):
            self.callbacks.extend(callback)
            return
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

        self.state.update_train(self.model, acc, cost, step)
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
        self.state.update_test(acc, cost)
        return acc, cost

    def train(self, train_loader, val_loader, optim, scheduler=None) -> None:
        # Train the model for epochs
        lr = self.config.lr
        self.state.set_lr(lr)
        for step in range(self.config.epoch):
            # train step
            self.__train_step(train_loader, optim, step)
            if scheduler is not None:
                lr = scheduler.get_last_lr()[0]
                self.state.set_lr(lr)
                scheduler.step()
            # test step
            self.__test_step(val_loader, step, self.config.epoch)
            # callback on step end
            for callback in self.callbacks:
                callback.on_step_end(self.state)
        # callback on train end
        for callback in self.callbacks:
            callback.on_train_end()

    def test(self, test_loader, model):
        # Test the model
        print("validate mode")
        test_acc, test_loss = self.__test_step(test_loader, 1, 1, model)
        print(f"test acc: {test_acc} loss: {test_loss}")
