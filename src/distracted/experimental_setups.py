import typing
from tabnanny import check

import torch.nn as nn
from transformers import EfficientNetForImageClassification, EfficientNetImageProcessor

import distracted.segmentation_nn as segmentation_nn
from distracted.adapters import get_adapter_model
from distracted.data_util import load_model

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
    dataset_kwargs: dict[str, any]
    params: dict  # Hyperparameters
    dataloader_kwargs: dict[str, any] = {}


def unpack(x):
    data_dict, label = x
    return [data_dict["pixel_values"].squeeze(), label]


def adapter_setup(**params) -> ExperimentSetup:
    adapters = params["adapters"]
    adapter_weight = params["adapter_weight"]
    model = get_adapter_model(
        MODEL_NAME, adapter_locations=adapters, adapter_weight=adapter_weight
    )

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
    ch = params["checkpoint"]
    model = load_model(ch) if ch else segmentation_nn.Net()
    if ch:
        params |= {"checkpoint": ch}

    return ExperimentSetup(
        model=model,
        model_name=f"{MODEL_NAME}_segment",
        dataset_kwargs={
            "returns": ["segment", "preprocessed_image", "label"],
            "transform": segmentation_nn.segment_transform,
            "fuck_your_ram": 1_000_000,
        },
        dataloader_kwargs={"collate_fn": segmentation_nn.collate_fn},
        params=params,
    )
