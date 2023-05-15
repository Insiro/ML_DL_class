from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torchvision import transforms, datasets, models
from torchmetrics import Accuracy

from trainer import Trainer
from utils import Config
from callback import SaveModelCallback

config = Config(epoch=10, lr=1e-4, batch_size=32)

# region customize model
model = models.resnet18(pretrained=True)
print(model)
model.fc = nn.Linear(512, 10)
model.fc.weight.requires_grad = True
model.fc.bias.requires_grad = True
params_to_update = [model.fc.weight, model.fc.bias]
print("fc weight requires grad: ")
print("fc bias requires grad: ")
model.to(config.device)
# endregion

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


optimizer = Adam(params_to_update, lr=config.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
trainer = Trainer(
    model,
    config,
    nn.CrossEntropyLoss(),
    Accuracy(task="multiclass", num_classes=10).to(config.device),
)

trainer.add_callback(SaveModelCallback(config, minimum_epoch=2))

trainer.train(train_loader, test_loader, optimizer, scheduler=scheduler)
