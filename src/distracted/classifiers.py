# https://github.com/pytorch/examples/blob/main/mnist/main.py
import time
from distracted.experimental_setups import (
    ExperimentSetup,
    adapter_setup,
    finetune_setup,
    segmentation_setup,
)

from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from distracted.data_util import DATA_PATH, timeit
from distracted.dataset_loader import DriverDataset
import mlflow
from mlflow import log_metric, log_params
import click
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from distracted.id2label import ID2LABEL


torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


@click.command()
@click.option("--batch-size", default=128)
@click.option("--model-name", default="google/efficientnet-b3")
@click.option("--adapters", default=str([3]))
@click.option("--top-lr", default=2.0, type=float)
@click.option("--top-decay", default=0.0, type=float)
@click.option("--body-lr", default=0.0, type=float)
@click.option("--body-decay", default=0.0, type=float)
@click.option("--gamma", default=1.0, type=float)
@click.option("--epochs", default=10)
@click.option("--adapter-weight", default=0.0, type=float)
def init_cli(
    batch_size=128,
    model_name="google/efficientnet-b3",
    adapters=str([3]),
    top_lr=2.0,
    top_decay=0.0,
    body_lr=0.0,
    body_decay=0.0,
    gamma=1.0,
    epochs=10,
    adapter_weight=0.0,
):
    adapters = eval(adapters)

    common_params = dict(
        batch_size=batch_size,
        model_name=model_name,
        top_lr=top_lr,
        top_decay=top_decay,
        gamma=gamma,
        epochs=epochs,
    )

    if adapters:
        setup = adapter_setup(**common_params, adapters=adapters, adapter_weight=adapter_weight)
    elif body_lr:
        setup = finetune_setup(**common_params, body_lr=body_lr, body_decay=body_decay)
    else:
        setup = segmentation_setup(**common_params)

    main(setup)


def cross_entropy_loss(output, labels):
    try:
        return F.cross_entropy(output.logits, labels)
    except AttributeError:
        return F.cross_entropy(output, labels)


def train(model, device, train_loader, optimizer, epoch, *, log_interval=10):
    model.train()
    train_loss = 0
    for batch_idx, (*data, target) in enumerate(train_loader):
        with timeit("start"):
            ...
        data = [d.to(device) for d in data]
        target = target.to(device)
        optimizer.zero_grad()
        output = model(*data)
        loss = cross_entropy_loss(output, target)
        loss.backward()
        train_loss += loss
        optimizer.step()
        with timeit("end"):
            ...
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss,
                )
            )
    return train_loss / len(train_loader)


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    # correct = 0
    with torch.no_grad():
        for *data, target in test_loader:
            data = [d.to(device) for d in data]
            target = target.to(device)
            output = model(*data)
            test_loss += cross_entropy_loss(output, target)
            # pred = output.logits.argmax(
            #     dim=1, keepdim=True
            # )  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()

    print(f"\nTest set Epoch {epoch}: Average loss: {test_loss:.4f}\n")
    # log_metric("val accuracy", correct / len(test_loader.dataset))
    return test_loss / len(test_loader)

def get_confusion_matrix(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        true_values = []
        predicted_values = []
        for *data, target in test_loader:
            data = [d.to(device) for d in data]
            target = target.to(device)
            output = model(*data)
            for row in output.logits:
                predicted_label = ID2LABEL[row.argmax(-1).item()]
                predicted_values.append(predicted_label)
            for row in target:
                true_label = ID2LABEL[row.item()]
                true_values.append(true_label)
        cm = confusion_matrix(true_values, predicted_values)
        df_cm = pd.DataFrame(cm, index = [i for i in ID2LABEL.values()],
                  columns = [i for i in ID2LABEL.values()])
        fig = plt.figure(figsize=(16,10))
        sns.heatmap(df_cm, annot=True, fmt='g')
    return fig



def get_optimiser_params(model, top_lr, top_decay, body_lr=0, body_decay=0, **_):
    top_params, body_params = [], []
    for n, p in model.named_parameters():
        params = top_params if "top" in n or "classifier" in n else body_params
        params.append(p)

    return [
        {"params": top_params, "lr": top_lr, "weight_decay": top_decay},
        {"params": body_params, "lr": body_lr, "weight_decay": body_decay},
    ]


def main(setup: ExperimentSetup):
    device = torch.device("cuda")

    data_kwargs = {
        # "num_workers": 4,
        "pin_memory": True,
        "shuffle": True,  # Only done once for entire run
        "batch_size": setup.params["batch_size"],
        "drop_last": True,  # Drop last batch if it's not full
    }

    train_loader, dev_loader = [
        DataLoader(
            DriverDataset(split, **setup.dataset_kwargs),
            **data_kwargs,
            **setup.dataloader_kwargs,
        )
        for split in ["train", "dev"]
    ]

    model = setup.model.to(device)

    optimizer = optim.Adadelta(get_optimiser_params(model, **setup.params))

    scheduler = StepLR(optimizer, step_size=1, gamma=setup.params["gamma"])

    mlflow.set_tracking_uri(uri=f"file://{DATA_PATH}\mlruns")

    best_test_loss = 999
    with mlflow.start_run():
        log_params(setup.params)

        for epoch in range(1, setup.params["epochs"] + 1):
            with timeit("train"):
                train_loss = train(
                    model, device, train_loader, optimizer, epoch, log_interval=10
                )
            with timeit("test"):
                if (
                    test_loss := test(model, device, dev_loader, epoch)
                ) < best_test_loss:
                    best_test_loss = test_loss
                    state = model.state_dict()

            scheduler.step()

            log_metric("train loss", train_loss)
            log_metric("val loss", test_loss)
        mlflow.pytorch.log_state_dict(state, "model")
        mlflow.pytorch.log_model(model, "model")
        fig = get_confusion_matrix(model, device, dev_loader)
        mlflow.log_figure(fig, "confusion_matrix.png")
        


if __name__ == "__main__":
    init_cli()
