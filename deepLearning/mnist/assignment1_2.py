from os import path
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchvision import transforms, datasets
from torchmetrics import Accuracy

from mobilenet import MobileNet
from trainer import Trainer
from callback import SaveModelCallback, TensorBoardWriter
from utils import Config


def main():
    config = Config(
        epoch=1000,
        lr=1e-3,
        batch_size=16,
        output_path="./output_mobilenet",
    )
    num_classes = 10

    torch.cuda.empty_cache()
    # region data loader
    train_set = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize([112, 112]),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
    test_set = datasets.MNIST(
        "./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize([112, 112]),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
    # create data loaders
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size)
    # endregion
    # create the model, optimizer, and trainer instance
    model = MobileNet(num_classes=num_classes)
    model.to(config.device)

    trainer = Trainer(
        model=model,
        config=config,
        loss_fn=CrossEntropyLoss(),
        acc_fn=Accuracy(task="multiclass", num_classes=num_classes).to(config.device),
    )
    trainer.add_callback([SaveModelCallback(config), TensorBoardWriter()])
    # train and test the model
    optimizer = torch.optim.Adadelta(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
    trainer.train(train_loader, test_loader, optim=optimizer, scheduler=scheduler)
    model.load_state_dict(torch.load(path.join(config.model_path, "best.pt")))
    trainer.test(test_loader, model)


main()
