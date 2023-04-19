# https://github.com/pytorch/examples/blob/main/mnist/main.py

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

B = BATCH_SIZE = 128  # For 12GB VRAM

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


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
        print(f"conv1: {x.shape=}")
        x = F.relu(x)

        x = self.conv2(x)
        print(f"conv2: {x.shape=}")
        x = F.relu(x)

        x = F.max_pool2d(x, 4)
        print(f"max_pool2d: {x.shape=}")
        x = self.dropout1(x)

        x = self.conv3(x)
        print(f"conv3: {x.shape=}")
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        print(f"max_pool2d: {x.shape=}")
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        print(f"flatten: {x.shape=}")

        x = self.fc1(x)
        print(f"fc1: {x.shape=}")
        x = F.relu(x)
        x = self.dropout2(x)
        x = torch.cat((x, embeddings), 1)
        x = self.fc2(x)
        print(f"fc2: {x.shape=}")

        output = F.log_softmax(x, dim=1)
        print(f"log_softmax: {output.shape=}")
        return output


def train(model, device, train_loader, optimizer, epoch, *, log_interval=10, embeddings):
    model.train()
    for batch_idx, (data, target, path_name) in enumerate(train_loader):
        path_names = [x[:-4] for x in path_name]
        embs = torch.tensor(embeddings.loc[path_names].values)
        data, target, embs = data.to(device), target.to(device), embs.to(device)
        optimizer.zero_grad()
        output = model(data, embs)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
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


def main():
    torch.manual_seed(42)

    LR = 1
    GAMMA = 0.7
    EPOCHS = 10
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
    embeddings = pd.read_parquet(DATA_PATH / "efficientnet_embeddings.parquet")

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=LR, rho=0, eps=0, weight_decay=0)

    scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)
    for epoch in range(1, EPOCHS + 1):
        train(model, device, segment_train_loader, optimizer, epoch, log_interval=10, embeddings=embeddings)
        test(model, device, segment_dev_loader)
        scheduler.step()

    if save_model := True:
        torch.save(model.state_dict(), "cnn.pt")


if __name__ == "__main__":
    main()
