# https://github.com/pytorch/examples/blob/main/mnist/main.py
import time

import click
import mlflow
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from mlflow import log_metric, log_params
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from distracted.data_util import DATA_PATH, ID2LABEL, timeit
from distracted.dataset_loader import DriverDataset
from distracted.experimental_setups import (
    ExperimentSetup,
    adapter_setup,
    finetune_setup,
    segmentation_setup,
)

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
@click.option("--checkpoint", default="", type=str)
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
    checkpoint="",
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
        setup = adapter_setup(
            **common_params, adapters=adapters, adapter_weight=adapter_weight
        )
    elif body_lr:
        setup = finetune_setup(**common_params, body_lr=body_lr, body_decay=body_decay)
    else:
        setup = segmentation_setup(**common_params, checkpoint=checkpoint)

    main(setup)


def cross_entropy_loss(output, labels):
    try:
        return F.cross_entropy(output, labels)
    except AttributeError:
        return F.cross_entropy(output, labels)


def train(model, device, train_loader, optimizer, epoch, *, log_interval=10):
    model.train()
    train_loss = 0
    for batch_idx, (*data, target) in enumerate(train_loader):
        data = [d.to(device) for d in data]
        target = target.to(device)
        optimizer.zero_grad()
        output = model(*data)
        loss = cross_entropy_loss(output, target)
        loss.backward()
        train_loss += loss
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data[0]),
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

    print(
        f"\nTest set Epoch {epoch}: Average loss: {(test_loss / len(test_loader)):.4f}\n"
    )
    # log_metric("val accuracy", correct / len(test_loader.dataset))
    return test_loss / len(test_loader)


def get_confusion_matrix(model, test_loader):
    model.eval()
    device = torch.device("cuda")
    with torch.no_grad():
        true_values = []
        predicted_values = []
        for *data, target in test_loader:
            data = [d.to(device) for d in data]
            target = target.to(device)
            output = model(*data)
            for row in output:
                predicted_label = ID2LABEL[row.argmax(-1).item()]
                predicted_values.append(predicted_label)
            for row in target:
                true_label = ID2LABEL[row.item()]
                true_values.append(true_label)
        num_identical = 0
        for i in range(len(true_values)):
            if true_values[i] == predicted_values[i]:
                num_identical += 1
        accuracy = num_identical / len(true_values)

        cm = confusion_matrix(true_values, predicted_values, normalize="true")
        df_cm = pd.DataFrame(
            cm,
            index=[i for i in ID2LABEL.values()],
            columns=[i for i in ID2LABEL.values()],
        )
        fig = plt.figure(figsize=(16, 12))
        sns.heatmap(df_cm, annot=True, fmt="g")
    return fig, accuracy


def get_optimiser_params(model, top_lr, top_decay, body_lr=0, body_decay=0, **_):
    top_params, body_params = [], []
    for n, p in model.named_parameters():
        params = top_params if "top" in n or "classifier" in n else body_params
        params.append(p)
    if not top_params:
        print("didnt find any top params! Assuming top_params = all_params")
        top_params = body_params
        body_params = []
        assert not body_lr and not body_decay

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
        try:
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

                scheduler.step()

                log_metric("train loss", train_loss, step=epoch)
                log_metric("val loss", test_loss, step=epoch)
        finally:
            mlflow.pytorch.log_model(model, "model")
            fig, _ = get_confusion_matrix(model, dev_loader)
            mlflow.log_figure(fig, "confusion_matrix.png")


if __name__ == "__main__":
    init_cli()
