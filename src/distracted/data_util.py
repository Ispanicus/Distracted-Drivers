import distracted
import torch
import pandas as pd
from PIL import Image
from pathlib import Path

DATA_PATH = Path(distracted.__path__[0]).parents[1] / "data"
assert DATA_PATH.exists()


H, W = 640, 480

MASK_LABELS = [
    "person",
    "bottle",
    "cell phone",
    "cup",
    "car",  # Idk if useful but lets grab it
    "chair",  # Idk if useful but lets grab it
]
C = len(MASK_LABELS)


class Tensor(torch.Tensor):
    def __class_getitem__(*args):
        ...  # allow Tensor[W, H, C]


def get_efficientnet_embeddings() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH / "efficientnet_embeddings.parquet")


def get_train_df():
    classes = {
        "c0": "safe driving",
        "c1": "texting - right",
        "c2": "talking on the phone - right",
        "c3": "texting - left",
        "c4": "talking on the phone - left",
        "c5": "operating the radio",
        "c6": "drinking",
        "c7": "reaching behind",
        "c8": "hair and makeup",
        "c9": "talking to passenger",
    }
    name_to_path = {f.name: f for f in DATA_PATH.rglob("*.jpg")}
    # test_files = [f for f in (DATA_PATH / 'imgs/test').rglob('*.jpg')]

    df = pd.read_csv(DATA_PATH / "driver_imgs_list.csv")
    df = df.assign(path=df.img.map(name_to_path), desc=df.classname.map(classes))

    df["img"] = [lambda row=row: Image.open(row.path) for row in df.itertuples()]
    return df
