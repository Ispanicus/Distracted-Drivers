import time
from distracted.data_util import (
    DATA_PATH,
    get_train_df,
    H,
    W,
    C,
    timeit,
    load_onehot,
)
import multiprocessing
import torchvision.io
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datasets import load_dataset
from typing import Literal
import pickle
from transformers import EfficientNetImageProcessor
from distracted.experimental_setups import PREPROCESSOR

# __df = get_train_df()
# __dev_subjects = (
#     __df.groupby("subject").path.count().sample(frac=0.15)
# )  # NOTE: The frac is not exact, as subjects have 346-1237 imgs each (most often ~800 though)
# __test_subjects = (
#     __df.groupby("subject").path.count().sample(frac=0.15)
# )
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


def dataset_loader():
    OBJ_PATH = DATA_PATH / "dataset_v2.obj"
    if OBJ_PATH.exists():
        with open(OBJ_PATH, "rb") as file:
            dataset = pickle.load(file)
            return dataset

    create_metadata()

    dataset = load_dataset(
        "imagefolder", data_dir=DATA_PATH / "imgs", drop_labels=False
    )

    with open(OBJ_PATH, "wb") as file:
        pickle.dump(dataset, file)

    return dataset


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

        # Load everything into memory :)
        i_know_what_im_about_to_do = str(fuck_your_ram)[-2:] == "42"
        if fuck_your_ram and not i_know_what_im_about_to_do:
            # memory *= n_processes
            self.naughty_boi = lambda: ...  # Crashes if multiprocessing
        self.fuck_your_ram = fuck_your_ram

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
            return load_onehot(
                next((DATA_PATH / "onehot").glob(f"{row.path.name}.npz"))
                )

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
        segment_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
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
        assert segments.shape == (BATCH_SIZE, H, W, C)
        assert labels.shape == (BATCH_SIZE,)
        break


def img_example():
    img_dataloader = dataset_loader()

    for data in img_dataloader["train"]:
        match data:
            case {
                "image": image,
                "label": label,
                "subject": subject,
                "classname": classname,
                "img": img_name,
            }:
                print(f"{image.size=} {label=} {subject=} {classname=} {img_name=}")

            case _:
                raise ValueError("Data did not follow expected format")
        break


if __name__ == "__main__":
    # segment_example()
    # img_example()

    from tqdm import tqdm
    from distracted.experimental_setups import preprocess_img

    BATCH_SIZE = 128
    with timeit("total"):
        with timeit("dataset"):
            segment_train_dataset = DriverDataset(
                "train",
                returns=["preprocessed_image", "label"],
                fuck_your_ram=1_000_042,
            )

        with timeit("dataloader"):
            segment_train_dataloader = DataLoader(
                segment_train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
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
