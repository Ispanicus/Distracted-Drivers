from distracted.data_util import DATA_PATH, get_train_df, H, W, C, Tensor
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from datasets import load_dataset, Features, Image, ClassLabel
from typing import Literal
from pathlib import Path
import pickle

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


class SegmentDataset(Dataset):
    def __init__(
        self,
        split: Literal["train"] | Literal["dev"] | Literal["test"],
        transform=None,
        target_transform=None,
    ):
        df = get_train_df()
        df = df.query(f"subject in @subjects[@split]")
        ok_files = set(df.path.map(lambda p: p.name.replace(".jpg", ".jpg.npz")))

        self.arr_paths = [
            p for p in (DATA_PATH / "onehot").glob("*.npz") if p.name in ok_files
        ]
        self.transform = transform
        self.target_transform = target_transform

        self.img_to_class = dict(
            zip(df.path.map(lambda p: p.name), df.classname.str[-1].astype(int))
        )

    def __len__(self):
        return len(self.arr_paths)

    def __getitem__(self, idx):
        from distracted.image_segmentation import load_onehot

        path = self.arr_paths[idx]
        tensor: Tensor[H, W, C] = load_onehot(path)
        label = self.img_to_class[path.name.replace(".jpg.npz", ".jpg")]
        if self.transform:
            tensor = self.transform(tensor)
        if self.target_transform:
            label = self.target_transform(label)
        return tensor, label


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
    segment_train_dataset = SegmentDataset("train", None, None)
    segment_train_dataloader = DataLoader(
        segment_train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    for train_features, train_labels in segment_train_dataloader:
        assert train_features.shape == (BATCH_SIZE, H, W, C)
        assert train_labels.shape == (BATCH_SIZE,)
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
    segment_example()
    img_example()
