from distracted.data_util import DATA_PATH
import pandas as pd
from datasets import load_dataset, Features, Image, ClassLabel
from pathlib import Path
import pickle


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
    pd.concat([meta, test_meta]).to_csv(DATA_PATH / "imgs/metadata.csv")


if __name__ == "__main__":
    dataset_loader()
