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


def preprocess_img(
    x,
):
    data, label = x
    preprocessed_data = PREPROCESSOR(data, return_tensors="pt")[
        "pixel_values"
    ].squeeze()
    return [preprocessed_data, label]


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
        dataloader_returns=["torch_image", "label"],
        transform=preprocess_img,
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
        dataloader_returns=["torch_image", "label"],
        transform=preprocess_img,
        params=params,
    )


def segmentation_setup(**params) -> ExperimentSetup:
    ...
