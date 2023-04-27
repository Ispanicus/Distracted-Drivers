import mlflow
import torch
from torch.utils.data import DataLoader

from distracted.classifiers import get_confusion_matrix
from distracted.data_util import load_model
from distracted.dataset_loader import DriverDataset
from distracted.experimental_setups import unpack

data_kwargs = {
    # "num_workers": 4,
    "pin_memory": True,
    "shuffle": True,  # Only done once for entire run
    "batch_size": 32,
    "drop_last": True,  # Drop last batch if it's not full
}
dataset_kwargs = {
    "returns": ["preprocessed_image", "label"],
    "fuck_your_ram": 1_000_042,
    "transform": unpack,
}
test_loader = DataLoader(
    DriverDataset("test", **dataset_kwargs),
    **data_kwargs,
)


def final_metrics(run_id: str, test_loader):
    device = torch.device("cuda")

    model_path = f"../data/mlruns/0/{run_id}/artifacts/model"
    model = mlflow.pytorch.load_model(model_path)
    fig, accuracy = get_confusion_matrix(model, test_loader)
    return fig, accuracy


if __name__ == "__main__":
    model = load_model("aa246d9d2106472492442ff362b1b143")
