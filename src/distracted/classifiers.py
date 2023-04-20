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
from distracted.dataset_loader import SegmentDataset
import mlflow

torch.manual_seed(1)

B = BATCH_SIZE = 64  # 128 For 12GB VRAM

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

from time import perf_counter
from contextlib import contextmanager
import datetime as dt


@contextmanager
def timeit(msg: str) -> float:
    start = perf_counter()
    start_date = f"{dt.datetime.now():%H:%M:%S}"
    yield
    print(f"{start_date} Time: {msg} {perf_counter() - start:.3f} seconds")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=C, out_channels=32, kernel_size=50, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 20, 2)
        self.conv3 = nn.Conv2d(64, 128, 5, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2304, 128)
        self.fc2 = nn.Linear(2688, 10)

    def forward(self, x, embeddings):
        print = lambda x: x  # Don't print
        x = self.conv1(x)
        print(f"conv1: {torch.mean(x)=}")
        x = F.relu(x)

        x = self.conv2(x)
        print(f"conv2: {torch.mean(x)=}")
        x = F.relu(x)

        x = F.max_pool2d(x, 4)
        print(f"max_pool2d: {torch.mean(x)=}")
        x = self.dropout1(x)

        x = self.conv3(x)
        print(f"conv3: {torch.mean(x)=}")
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        print(f"max_pool2d: {torch.mean(x)=}")
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        print(f"flatten: {torch.mean(x)=}")

        x = self.fc1(x)
        print(f"fc1: {torch.mean(x)=}")
        x = F.relu(x)
        x = self.dropout2(x)
        print(f"embeddings: {torch.mean(embeddings)=}")
        x = torch.cat((x, embeddings), 1)
        x = self.fc2(x)
        print(f"fc2: {torch.mean(x)=}")

        output = F.log_softmax(x, dim=1)
        print(f"log_softmax: {torch.mean(output)=}")
        return output


def train(
    model, device, train_loader, optimizer, epoch, *, log_interval=10, embeddings
):
    model.train()
    for batch_idx, (data, target, path_name) in enumerate(train_loader):
        path_names = [x[:-4] for x in path_name]
        embs = torch.tensor(embeddings.loc[path_names].values, requires_grad=True)
        data, target, embs = data.to(device), target.to(device), embs.to(device)
        optimizer.zero_grad()
        output = model(data, embs)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        mlflow.log_metric("loss", loss.item())
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


def test(model, device, test_loader, embeddings):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, path_name in test_loader:
            path_names = [x[:-4] for x in path_name]
            embs = torch.tensor(embeddings.loc[path_names].values, requires_grad=True)
            data, target, embs = data.to(device), target.to(device), embs.to(device)
            output = model(data, embs)
            test_loss += F.nll_loss(
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


@functools.cache
def load_embeddings():
    embeddings = pd.read_parquet(DATA_PATH / "efficientnet_embeddings.parquet")
    return embeddings


def main():
    torch.manual_seed(42)

    LR = 1
    GAMMA = 0.7
    EPOCHS = 1
    device = torch.device("cuda")

    data_kwargs = {
        "num_workers": 4,
        "pin_memory": True,
        "shuffle": True,
        "batch_size": BATCH_SIZE,
        "drop_last": True,  # Drop last batch if it's not full
    }

    transform = transforms.Compose([permute, transforms.Normalize((0,), (1,))])

    segment_train_loader = DataLoader(
        SegmentDataset("train", transform=transform), **data_kwargs
    )
    segment_dev_loader = DataLoader(
        SegmentDataset("dev", transform=transform), **data_kwargs
    )
    embeddings = load_embeddings()

    model = Net().to(device)
    optimizer = optim.Adadelta(
        model.parameters(),
        lr=LR,
    )  # rho=0, eps=0, weight_decay=0)

    scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)

    mlflow.set_tracking_uri(uri=f"file://{DATA_PATH}\mlruns")

    with mlflow.start_run():
        state_dict = model.state_dict()
        mlflow.pytorch.log_state_dict(state_dict, artifact_path="model")
        mlflow.log_params({"lr": LR, "gamma": GAMMA, "epochs": EPOCHS})

        for epoch in range(1, EPOCHS + 1):
            train(
                model,
                device,
                segment_train_loader,
                optimizer,
                epoch,
                log_interval=10,
                embeddings=embeddings,
            )
            test(model, device, segment_dev_loader, embeddings=embeddings)

            scheduler.step()

        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    main()
