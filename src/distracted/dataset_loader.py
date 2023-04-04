import distracted
from datasets import load_dataset, Features, Image, ClassLabel
from pathlib import Path
import pickle


def dataset_creator():
    features = Features(
        {
            "label": ClassLabel(
                num_classes=10,
                names=["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"],
            ),
            "image": Image(),
        }
    )

    DATA_PATH = Path(distracted.__path__[0]).parents[1] / "data/imgs"
    assert DATA_PATH.exists()

    dataset = load_dataset("imagefolder", data_dir= DATA_PATH, drop_labels=False)
    return dataset
    
def dataset_loader():
    HOME_PATH = Path(distracted.__path__[0]).parents[1] / "dataset.obj"
    file = open(HOME_PATH,'rb')
    dataset = pickle.load(file)
    file.close()
    return dataset