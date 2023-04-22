# https://github.com/pytorch/examples/blob/main/mnist/main.py
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
def init_cli(
    batch_size,
    model_name,
    adapters,
    top_lr,
    top_decay,
    body_lr,
    body_decay,
    gamma,
    epochs,
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
        setup = adapter_setup(**common_params, adapters=adapters)
    elif body_lr:
        setup = finetune_setup(**common_params, body_lr=body_lr, body_decay=body_decay)
    else:
        setup = segmentation_setup(**common_params)

    main(setup)


def train(model, device, train_loader, optimizer, epoch, *, log_interval=10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output.logits, target)
        loss.backward()
        optimizer.step()
        log_metric("train loss", loss.item(), batch_idx + len(train_loader) * epoch)
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


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
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
    return test_loss


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
        # "pin_memory": True,
        "shuffle": True,
        "batch_size": setup.params["batch_size"],
        "drop_last": True,  # Drop last batch if it's not full
    }

    train_loader, dev_loader = [
        DataLoader(
            DriverDataset(split, **setup.dataset_kwargs),
            **data_kwargs,
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
            train(model, device, train_loader, optimizer, epoch, log_interval=10)
            if (test_loss := test(model, device, dev_loader, epoch)) < best_test_loss:
                best_test_loss = test_loss
                state = model.state_dict()

            scheduler.step()

        mlflow.pytorch.log_state_dict(state, "model")
        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    init_cli()
