# https://github.com/pytorch/examples/blob/main/mnist/main.py

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from distracted.data_util import DATA_PATH, get_train_df, H, W, C, Tensor
from distracted.dataset_loader import DriverDataset
from distracted.adapters import get_adapter_model
from transformers import EfficientNetForImageClassification, EfficientNetImageProcessor
import mlflow
from mlflow import log_metric, log_params
import click


<<<<<<< HEAD
torch.manual_seed(42)
=======
B = BATCH_SIZE = 32 # 128 For 12GB VRAM

#MODEL_NAME = "google/efficientnet-b0"
MODEL_NAME = "google/efficientnet-b3"
MODEL_NAME_SUFFIX = MODEL_NAME[-2:]
USE_ADAPTER = True
MODEL_NAME_MLFOW = MODEL_NAME + "_adapter" if USE_ADAPTER else MODEL_NAME
ADAPTERS=[(3,7)]
LR = 2
GAMMA = 1
EPOCHS = 10
global best_test_loss
best_test_loss = 9999
global sd
global m
torch.manual_seed(42)

preprocessor = EfficientNetImageProcessor.from_pretrained(MODEL_NAME)

>>>>>>> a7309608c981626ab1b8e9d2fac7dac703ad58bd
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


@click.command()
@click.option("--batch-size", default=128)
@click.option("--model-name", default="google/efficientnet-b3")
@click.option("--adapters", default=str([(3, 7)]))
@click.option("--lr", default=2)
@click.option("--gamma", default=1)
@click.option("--epochs", default=10)
def init_cli(batch_size, model_name, adapters, lr, gamma, epochs):
    global BATCH_SIZE, MODEL_NAME, ADAPTERS, LR, GAMMA, EPOCHS, USE_ADAPTER, MODEL_NAME_SUFFIX, MODEL_NAME_MLFOW, preprocessor, best_test_loss
    BATCH_SIZE, MODEL_NAME, ADAPTERS, LR, GAMMA, EPOCHS = (
        batch_size,
        model_name,
        adapters,
        lr,
        gamma,
        epochs,
    )
    ADAPTERS = eval(ADAPTERS)
    USE_ADAPTER = bool(ADAPTERS)
    MODEL_NAME_SUFFIX = MODEL_NAME[-2:]
    MODEL_NAME_MLFOW = MODEL_NAME + "_adapter" if USE_ADAPTER else MODEL_NAME
    best_test_loss = 9999
    preprocessor = EfficientNetImageProcessor.from_pretrained(MODEL_NAME)
    main()


def train(model, device, train_loader, optimizer, epoch, *, log_interval=10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        len_data = len(data)
        data = preprocessor(data, return_tensors="pt")
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(**data)
        loss = F.cross_entropy(output.logits, target)
        loss.backward()
        optimizer.step()
        log_metric("train loss", loss.item(), batch_idx + len(train_loader) * epoch)
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len_data,
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = preprocessor.preprocess(data, return_tensors="pt")
            data, target = data.to(device), target.to(device)
            output = model(**data)
            log_metric(
                "val loss",
                F.cross_entropy(output.logits, target).item(),
                batch_idx + len(test_loader) * epoch,
            )
            test_loss += F.cross_entropy(
                output.logits, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.logits.argmax(
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
    log_metric("val accuracy", correct / len(test_loader.dataset))

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        sd = model.state_dict()
        m = model
    if epoch == EPOCHS:
        mlflow.pytorch.log_state_dict(sd, "model")
        mlflow.pytorch.log_model(m, "model")


def main():
    device = torch.device("cuda")

    data_kwargs = {
        "num_workers": 4,
        "pin_memory": True,
        "shuffle": True,
        "batch_size": BATCH_SIZE,
        "drop_last": True,  # Drop last batch if it's not full
    }

    train_loader = DataLoader(
        DriverDataset("train", returns=["torch_image", "label"]),
        **data_kwargs,
    )

    dev_loader = DataLoader(
        DriverDataset("dev", returns=["torch_image", "label"]),
        **data_kwargs,
    )

    if USE_ADAPTER:
        model = get_adapter_model(MODEL_NAME_SUFFIX, adapter_locations=ADAPTERS)
    else:
        model = EfficientNetForImageClassification.from_pretrained(MODEL_NAME)

    num_classes = 10
    model.classifier = nn.Linear(model.config.hidden_dim, num_classes)
    model.to(device)

    for name, param in model.named_parameters():
        grad: bool = any(label in name for label in ["classifier", "top", "adapters"])
        param.requires_grad = grad

    optimizer = optim.Adadelta(
        model.parameters(),
        lr=LR,
    )  # rho=0, eps=0, weight_decay=0)

    scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)

    mlflow.set_tracking_uri(uri=f"file://{DATA_PATH}\mlruns")

    with mlflow.start_run():
        log_params(
            {
                "lr": LR,
                "gamma": GAMMA,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "model_name": MODEL_NAME_MLFOW,
                "adapters": ADAPTERS,
            }
        )

        for epoch in range(1, EPOCHS + 1):
            train(model, device, train_loader, optimizer, epoch, log_interval=10)
            test(model, device, dev_loader, epoch)

            scheduler.step()


if __name__ == "__main__":
    init_cli()
