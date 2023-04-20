# https://github.com/pytorch/examples/blob/main/mnist/main.py

import functools
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from distracted.data_util import DATA_PATH, get_train_df, H, W, C, Tensor
from distracted.dataset_loader import DriverDataset
from transformers import EfficientNetForImageClassification
import mlflow
from mlflow import log_metric, log_metrics, log_params, log_artifacts, set_tracking_uri, set_experiment

B = BATCH_SIZE = 64  # 128 For 12GB VRAM

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

def train(model, device, train_loader, optimizer, epoch, *, log_interval=10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        log_metric("loss", loss.item())
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def permute(x: Tensor[H, W, C]) -> Tensor:
    x: Tensor[C, H, W] = x.permute(2, 0, 1)
    return x

def main():
    torch.manual_seed(42)

    LR = 0.01
    GAMMA = 1
    EPOCHS = 1
    device = torch.device("cuda")

    data_kwargs = {
        "num_workers": 4,
        "pin_memory": True,
        "shuffle": True,
        "batch_size": BATCH_SIZE,
        "drop_last": True,  # Drop last batch if it's not full
    }

    preprocessor = EfficientNetImageProcessor.from_pretrained('google/efficientnet-b0')

    def transform_images(x):
        images, labels = x
        x[0] = preprocessor(images, return_tensors="pt")
        return x 
    
    train_loader = DataLoader(
        DriverDataset("train", returns=["torch_image","label"], transform=transform_images), **data_kwargs
    )
    dev_loader = DataLoader(
        DriverDataset("dev", returns=["torch_image","label"], transform=transform_images), **data_kwargs
    )
    
    model = EfficientNetForImageClassification.from_pretrained('google/efficientnet-b0')

    # Set requires_grad to False for all layers except the last two blocks
    for param in model.parameters():
        param.requires_grad = False

    config = model.config
    num_classes = 10
    model.classifier = nn.Linear(config.hidden_dim, num_classes)

    for param in model.classifier.parameters():
        param.requires_grad = True
    for name, param in model.named_parameters():
        if 'top' in name:
            param.requires_grad = True
            print('top layer found')

    model.to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=LR, )#rho=0, eps=0, weight_decay=0)

    scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)

    mlflow.set_tracking_uri(uri=f'file://{DATA_PATH}\mlruns')

    with mlflow.start_run():

        artifact_path = "model"
        state_dict = model.state_dict()

        log_params({"lr": LR, "gamma": GAMMA, "epochs": EPOCHS})

        for epoch in range(1, EPOCHS + 1):
            train(model, device, train_loader, optimizer, epoch, log_interval=10)
            test(model, device, dev_loader)
            
            scheduler.step()
        mlflow.pytorch.log_state_dict(state_dict, artifact_path)
        mlflow.pytorch.log_model(model, artifact_path)
    

if __name__ == "__main__":
    main()
