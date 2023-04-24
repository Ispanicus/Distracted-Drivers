import functools
import time
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from distracted.data_util import DATA_PATH, H, W, C, Tensor, timeit

D = 2560  # Embedding dimension


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=16, kernel_size=16),
            nn.ReLU(),
            nn.MaxPool2d(8),
            nn.Conv2d(16, 8, 16),
            nn.ReLU(),
            nn.MaxPool2d(8),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(280, 128),
            nn.ReLU(),
        )
        self.seq2 = nn.Sequential(  # new seq to allow concat embedding
            nn.Linear(D + 128, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, segment, imagenet_embeddings):
        x = self.seq1(segment)
        x = torch.cat((x, imagenet_embeddings), 1)
        x = self.seq2(x)
        return x


def permute(x: Tensor[H, W, C]) -> Tensor:
    x: Tensor[C, H, W] = x.permute(2, 0, 1)
    return x


@functools.cache
def load_embeddings():
    embeddings = pd.read_parquet(DATA_PATH / "efficientnet_embeddings.parquet")
    return embeddings


def dummy_imagenet(x: Tensor[1, 3, H, W]) -> Tensor:
    one, three, w, h = x.shape
    assert one == 1 and three == 3, f"Expected (1, 3, {w}, {h}) but got {x.shape}"
    return torch.ones((D,), dtype=torch.float32)


def collate_fn(
    x: list[tuple[Tensor[H, W, C], Tensor[3, H, W], int]]
) -> tuple[Tensor["B", H, W, C], Tensor["B", 3, H, W], list[int]]:
    """Turn list[Tensor[x,y,z]] -> Tensor[Batch,x,y,z]"""
    segment, preprocessed_img, label = zip(*x)
    segment = torch.stack(segment)
    preprocessed_img = torch.stack(preprocessed_img)
    return segment, preprocessed_img, torch.tensor(label)


def segment_transform(
    x: tuple[Tensor["B", H, W, C], Tensor["B", 3, H, W], list[int]]
) -> Tensor:
    segment, preprocessed_img, label = x
    transform = transforms.Compose([permute, transforms.Normalize((0,), (1,))])
    segment = transform(segment)
    # assert len(preprocessed_img["pixel_values"]) == 1
    imagenet_embeddings = dummy_imagenet(preprocessed_img["pixel_values"])
    return segment, imagenet_embeddings, label


if __name__ == "__main__":
    ...
