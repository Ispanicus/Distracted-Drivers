import datetime as dt
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image

import distracted

DATA_PATH = Path(distracted.__path__[0]).parents[1] / "data"
assert DATA_PATH.exists()


H, W = 480, 640

MASK_LABELS = [
    "person",
    "bottle",
    "cell phone",
    "cup",
    "car",  # Idk if useful but lets grab it
    "chair",  # Idk if useful but lets grab it
]
C = len(MASK_LABELS)

CLASS_LABELS = [
    "normal driving",
    "texting - right",
    "talking on the phone - right",
    "texting - left",
    "talking on the phone - left",
    "operating the radio",
    "drinking",
    "reaching behind",
    "hair and makeup",
    "talking to passenger",
]

ID2LABEL = dict(enumerate(CLASS_LABELS))
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


class Tensor(torch.Tensor):
    def __class_getitem__(*args):
        ...  # allow Tensor[H, W, C]


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


@contextmanager
def timeit(msg: str) -> float:
    start = perf_counter()
    start_date = f"{dt.datetime.now():%H:%M:%S}"
    yield
    print(f"{start_date} Time: {msg} {perf_counter() - start:.3f} seconds")


def save_onehot(path: Path, onehot: Tensor[H, W, C]):
    np.savez_compressed(path, data=onehot.to(bool).numpy())


def load_onehot(path: Path) -> Tensor[H, W, C]:
    return torch.from_numpy(np.load(path)["data"])


def load_model(id: str):
    """E.g.: model = load_model("aa246d9d2106472492442ff362b1b143")"""
    return mlflow.pytorch.load_model(
        model_uri=f"file://{DATA_PATH}/mlruns/0/{id}/artifacts/model"
    )
