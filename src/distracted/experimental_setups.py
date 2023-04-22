import typing
import torch.nn as nn
from distracted.adapters import get_adapter_model
from transformers import EfficientNetForImageClassification, EfficientNetImageProcessor

MODEL_NAME = "google/efficientnet-b3"
PREPROCESSOR = EfficientNetImageProcessor.from_pretrained(
    MODEL_NAME
)  # load globally to avoid .2s overhead per batch


class ExperimentSetup(typing.NamedTuple):
    """
    Encapsulate all model specific information into a single object.
    May have to extend it with model specific hyperparams?
    """

    model: nn.Module  # Accepts a batch of data and spits out class probabilities
    model_name: str  # Name of model
    dataloader_returns: list[str]  # E.g. ["segment", "torch_image", "label"]
    transform: callable  # Applied in dataloader before data is fed to model
    params: dict  # Hyperparameters


def unpack(x):
    data_dict, label = x
    return [data_dict["pixel_values"].squeeze(), label]


def adapter_setup(**params) -> ExperimentSetup:
    adapters = params["adapters"]
    model = get_adapter_model(MODEL_NAME, adapter_locations=adapters)

    num_classes = 10
    model.classifier = nn.Linear(model.config.hidden_dim, num_classes)

    for name, param in model.named_parameters():
        grad: bool = any(label in name for label in ["classifier", "top", "adapters"])
        param.requires_grad = grad

    return ExperimentSetup(
        model=model,
        model_name=f"{MODEL_NAME}_adapter",
        dataset_kwargs={
            "returns": ["preprocessed_image", "label"],
            "fuck_your_ram": 1_000_042,
            "transform": unpack,
        },
        params=params,
    )


def finetune_setup(**params) -> ExperimentSetup:
    """TODO: Finetune entire model, not just last layer"""
    model = EfficientNetForImageClassification.from_pretrained(MODEL_NAME)

    num_classes = 10
    model.classifier = nn.Linear(model.config.hidden_dim, num_classes)

    for name, param in model.named_parameters():
        grad: bool = any(label in name for label in ["classifier", "top"])
        param.requires_grad = grad

    return ExperimentSetup(
        model=model,
        model_name=f"{MODEL_NAME}_finetune",
        dataset_kwargs={
            "returns": ["preprocessed_image", "label"],
            "fuck_your_ram": 1_000_042,
            "transform": unpack,
        },
        params=params,
    )


def segmentation_setup(**params) -> ExperimentSetup:
    ...
