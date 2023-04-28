import operator

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from mirror.visualisations.core import GradCam
from PIL import Image
from transformers import EfficientNetImageProcessor

from distracted.data_util import get_train_df, load_model

MODEL_NAME = "google/efficientnet-b3"
PREPROCESSOR = EfficientNetImageProcessor.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pp(x):
    x = x - x.min()
    x = x / x.max()
    return x.clamp(0, 1)


def get_predictions(
    model, classname, subject, display_check: callable = lambda pred, true: pred == true
):
    cam = GradCam(model, device=device)
    image_funcs: list[callable] = (
        get_train_df().query("subject == @subject and classname == @classname").img
    )

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    j = 0
    for image_func in image_funcs:
        if j >= 10:
            break

        image = PREPROCESSOR(image_func(), return_tensors="pt")[
            "pixel_values"
        ].squeeze()
        output = model(image.unsqueeze(0).to(device))

        pred_class = output.argmax()
        if not display_check(pred_class, int(classname[-1])):
            continue

        tensr = cam(
            input_image=image.unsqueeze(0).to(device), layer=None, postprocessing=pp
        )[0].squeeze(0)
        img = Image.fromarray(np.array(255 * tensr.permute(1, 2, 0)).astype(np.uint8))

        row, col = divmod(j, 5)
        axes[row, col].imshow(img)
        axes[row, col].axis("off")
        axes[row, col].set_title(f"True: {classname} \n Predicted {pred_class}")
        j += 1
    plt.show()


def confused_predictions(model, classname="c0", subject="p026"):
    get_predictions(model, classname, subject, display_check=operator.ne)


def correct_predictions(model, classname="c0", subject="p026"):
    get_predictions(model, classname, subject, display_check=operator.eq)


if __name__ == "__main__":
    model = load_model("d80d9fd0c2d849c1bee77bbe87c95566")
    confused_predictions(model)
    correct_predictions(model)
