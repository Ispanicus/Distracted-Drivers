import functools

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from distracted.data_util import DATA_PATH, C, H, Tensor, W, timeit

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
        )
        self.seq2 = nn.Sequential(  # new seq to allow concat embedding
            nn.Linear(D + 280, 10),
            nn.Dropout(0.25),
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


def dummy_imagenet(x: Tensor["B", 3, H, W]) -> Tensor:
    one, three, w, h = x.shape
    assert one == 1 and three == 3, f"Expected (1, 3, {w}, {h}) but got {x.shape}"
    return torch.ones(
        (D,),
        dtype=torch.float32,
    )


def collate_fn(
    x: list[tuple[Tensor[H, W, C], Tensor[3, H, W], int]]
) -> tuple[Tensor["B", H, W, C], Tensor["B", 3, H, W], list[int]]:
    """Turn list[Tensor[x,y,z]] -> Tensor[Batch,x,y,z]"""
    segment, embeddings, label = zip(*x)
    # Turn class_arr (0, 1, 2, etc.) into one_hot
    class_arr = torch.stack(segment).to(torch.int64)
    one_hot = F.one_hot(class_arr, num_classes=C).to(torch.float32).permute(0, 3, 1, 2)
    return one_hot, torch.stack(embeddings), torch.tensor(label)


def segment_transform(x: tuple[Tensor[H, W, C], Tensor[3, H, W], list[int]]) -> Tensor:
    segment, preprocessed_img, label = x
    imagenet_embeddings = dummy_imagenet(preprocessed_img["pixel_values"])
    return segment, imagenet_embeddings, label


if __name__ == "__main__":
    ...
