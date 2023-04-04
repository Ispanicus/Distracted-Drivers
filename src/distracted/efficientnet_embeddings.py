import torch
import pandas as pd
import numpy as np
from transformers import (
    EfficientNetImageProcessor,
    EfficientNetForImageClassification,
    EfficientNetConfig,
)
from PIL import Image
from distracted.data_util import DATA_PATH
from distracted.dataset_loader import dataset_loader

config = EfficientNetConfig.from_pretrained("google/efficientnet-b7")
config.num_labels = 0
model = EfficientNetForImageClassification(config).to("cuda")

preprocessor = EfficientNetImageProcessor.from_pretrained("google/efficientnet-b7")
dataset = dataset_loader()

img_embeddings = []
for sample in dataset["train"]:
    img = sample["image"]
    inputs = preprocessor(img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**{k: v.to("cuda") for k, v in inputs.items()}).logits
    img_embeddings.append(logits.cpu().numpy())

df = pd.DataFrame(np.vstack(img_embeddings), columns=map(str, range(2560)))
df.to_parquet(DATA_PATH / "efficientnet_embeddings.parquet")
