from typing import Literal

import pandas as pd
import torchvision.io
from torch.utils.data import DataLoader, Dataset

from distracted.data_util import DATA_PATH, C, H, W, get_train_df, load_onehot, timeit
from distracted.experimental_setups import PREPROCESSOR

# Sizes: dev: 3741, test: 3526, train: 15157
subjects = {
    "dev": ["p051", "p050", "p047", "p026"],
    "test": ["p066", "p049", "p014", "p041"],
    "train": [
        "p002",
        "p012",
        "p015",
        "p016",
        "p021",
        "p022",
        "p024",
        "p035",
        "p039",
        "p042",
        "p045",
        "p052",
        "p056",
        "p061",
        "p064",
        "p072",
        "p075",
        "p081",
    ],
}


class DriverDataset(Dataset):
    def __init__(
        self,
        split: Literal["train"] | Literal["dev"] | Literal["test"],
        returns=["img_name", "torch_image", "preprocessed_image", "segment", "label"],
        transform=None,
        *,
        fuck_your_ram: int = 0,
    ):
        self.transform = transform
        self.returns = returns
        self.df = (
            get_train_df().query(f"subject in @subjects[@split]").drop(columns="img")
        )

        self.fuck_your_ram = fuck_your_ram  # Load this many samples into memory

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if getattr(self, "_gigacache", None) is None:
            self._gigacache = []  # avoid infinite loop
            self._gigacache = [
                self[i] for i in range(min(len(self), self.fuck_your_ram))
            ]  # self[i] is important: __iter__ == __bad__

        if idx < len(self._gigacache):
            return self._gigacache[idx]

        row = self.df.iloc[idx]

        def load_img(row):
            return torchvision.io.read_image(str(row.path.absolute()))

        def load_segment(row):
            return load_onehot(next(DATA_PATH.glob(f"onehot/{row.path.name}.npz")))

        def preprocess_img(row):
            return PREPROCESSOR(load_img(row), return_tensors="pt")

        # Do callables, such that we do lazy loading
        returns = {
            "img_name": lambda row: row.path.name,
            "torch_image": load_img,
            "preprocessed_image": preprocess_img,
            "segment": load_segment,
            "label": lambda row: int(row.classname[1:]),
        }

        output = [returns[key](row) for key in self.returns]
        if self.transform:
            output = self.transform(output)
        return output


def create_metadata():
    meta = pd.read_csv(DATA_PATH / "driver_imgs_list.csv")
    meta = meta.assign(file_name="train/" + meta.classname + "/" + meta.img)
    assert len(meta.columns) == 4
    test_meta = pd.DataFrame(
        [
            ["?", "?", p.name, f"test/{p.name}"]
            for p in DATA_PATH.glob("imgs/test/*.jpg")
        ],
        columns=meta.columns,
    )
    pd.concat([meta, test_meta]).to_csv(DATA_PATH / "imgs/metadata.csv", index=False)


def segment_example():
    BATCH_SIZE = 4
    segment_train_dataset = DriverDataset(
        "train",
        returns=["img_name", "torch_image", "preprocessed_image", "segment", "label"],
    )
    segment_train_dataloader = DataLoader(
        segment_train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    for (
        img_names,
        torch_images,
        preprocessed,
        segments,
        labels,
    ) in segment_train_dataloader:
        assert len(img_names) == BATCH_SIZE
        assert torch_images.shape == (BATCH_SIZE, 3, H, W)
        assert "pixel_values" in preprocessed
        assert segments.shape == (BATCH_SIZE, H, W)
        assert labels.shape == (BATCH_SIZE,)
        break


if __name__ == "__main__":
    segment_example()

    from tqdm import tqdm

    from distracted.experimental_setups import PREPROCESSOR
    from distracted.segmentation_nn import collate_fn, segment_transform

    BATCH_SIZE = 128
    with timeit("total"):
        with timeit("dataset"):
            segment_train_dataset = DriverDataset(
                split="train",
                returns=["segment", "preprocessed_image", "label"],
                transform=segment_transform,
                fuck_your_ram=128,
            )

        with timeit("dataloader"):
            segment_train_dataloader = DataLoader(
                segment_train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
                collate_fn=collate_fn,
            )

        with timeit("iter dataloader"):
            iterator = iter(segment_train_dataloader)

        with timeit("loop"):
            for torch_images, labels in tqdm(iterator):
                pass

        with timeit("iter dataloader"):
            iterator = iter(segment_train_dataloader)

        with timeit("loop w. cuda"):
            for torch_images, labels in tqdm(iterator):
                torch_images.to("cuda")
