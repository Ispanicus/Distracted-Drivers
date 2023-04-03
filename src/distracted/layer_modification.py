import torch
from transformers import (
    EfficientNetImageProcessor,
    EfficientNetForImageClassification,
    EfficientNetConfig,
)
from PIL import Image
from distracted.data_util import DATA_PATH

image = Image.open(DATA_PATH / "imgs/train/c0/img_34.jpg")

config = EfficientNetConfig.from_pretrained("google/efficientnet-b7")
config.num_labels = 0
model = EfficientNetForImageClassification(config)

model.save_pretrained()
sorted(model.state_dict().keys())
model.get_submodule("classifier").state_dict()

logits.shape
model.num_hidden_layers
w = model.get_submodule("classifier").weight
w.shape
before_clf = logits @ w
before_clf.shape
dir(model)
model.state_dict()
model.modules()
classifier


# # model predicts one of the 1000 ImageNet classes
# predicted_label = logits.argmax(-1).item()
# print(model.config.id2label[predicted_label]),

model
