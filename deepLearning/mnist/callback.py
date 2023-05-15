from typing import Any
import os
import json
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils import Config


class StepState:
    """class for saving the state of the model at each step"""

    def __init__(self) -> None:
        self.train_acc = 0
        self.test_acc = 0
        self.train_loss = 0
        self.test_loss = 0
        self.step = 0
        self.lr = 0

    def update_train(self, model, train_acc, train_loss, step):
        self.model = model
        self.train_acc = train_acc
        self.train_loss = train_loss
        self.step = step

    def set_lr(self, lr):
        self.lr = lr

    def update_test(self, test_acc, test_loss):
        self.test_acc = test_acc
        self.test_loss = test_loss

    def update(self, model, train_acc, train_loss, test_acc, test_loss, step, lr):
        self.update_train(model, train_acc, train_loss, step, lr)
        self.update_test(test_acc, test_loss)


class CallbackBase:
    """Interface for callbacks"""

    def on_train_begin(self) -> Any:
        pass

    def on_train_end(self) -> Any:
        pass

    def on_epoch_begin(self, state: StepState) -> Any:
        pass

    def on_epoch_end(self, state: StepState) -> Any:
        pass

    def on_step_begin(self, state: StepState) -> Any:
        pass

    def on_step_end(self, state: StepState) -> Any:
        pass


class SaveModelCallback(CallbackBase):
    def __init__(self, config: Config, minimum_epoch: int = 100) -> None:
        # Initialize variables for saving the best model and results
        self.best_loss: Any = np.inf
        self.save_path = config.output_path
        self.minimum_epoch = minimum_epoch
        self.acc: Any = 0
        self.best_epoch: Any = np.inf

    def on_step_end(self, state: StepState) -> None:
        # Save the model at the end of every epoch, and the best model based on the loss function
        step = state.step
        if step < self.minimum_epoch:
            return
        # determine save file name
        output_name = f"{step}_{state.test_loss}_{state.test_acc}.pth"
        torch.save(state.model.state_dict(), os.path.join(self.save_path, output_name))
        if state.test_loss < self.best_loss:
            self.acc = state.test_acc
            self.best_loss = state.test_loss
            self.best_epoch = step
            torch.save(
                state.model.state_dict(),
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


class TensorBoardWriter(CallbackBase):
    def __init__(self) -> None:
        # Initialize the TensorBoard writer
        self.tb_writer = SummaryWriter()

    def on_step_end(self, state) -> Any:
        step = state.step
        self.tb_writer.add_scalar(f"Loss/train_loss", state.train_loss, step)
        self.tb_writer.add_scalar(f"Acc/train_ccc", state.train_acc, step)
        self.tb_writer.add_scalar(f"lr", state.lr, step)
        self.tb_writer.add_scalar(f"Loss/test_loss", state.test_loss, step)
        self.tb_writer.add_scalar(f"Acc/test_acc", state.test_acc, step)

        print(f"train acc: {state.train_acc} loss: {state.train_loss}, lr : {state.lr}")
        print(f"test acc: {state.test_acc} loss: {state.test_loss}")

    def on_train_end(self):
        self.tb_writer.flush()
