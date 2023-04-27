from mirror.visualisations.core import GradCam
import matplotlib.pyplot as plt
import torch
from PIL import Image
import mlflow
from distracted.data_util import get_train_df, load_model
from transformers import EfficientNetImageProcessor
import numpy as np

MODEL_NAME = "google/efficientnet-b3"
PREPROCESSOR = EfficientNetImageProcessor.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pp(x):
    x = x - x.min()
    x = x / x.max()
    return (x).clamp(0, 1)


def preprocess_images(images):
    preprocessed_images = []
    for image_func in images:
        preprocessed_images.append(
            PREPROCESSOR(image_func(), return_tensors="pt")["pixel_values"].squeeze()
        )
    return preprocessed_images


def correct_predictions(model, classname="c0", subject="p026"):
    cam = GradCam(model, device=device)
    df = get_train_df()
    images = df.query("subject == @subject and classname == @classname")["img"]
    preprocessed_images = preprocess_images(images)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    j = 0
    for image in preprocessed_images:
        if j >= 10:
            break
        output = model(image.unsqueeze(0).to(device))
        if output.argmax() == int(classname[-1]):
            tensr = cam(
                input_image=image.unsqueeze(0).to(device), layer=None, postprocessing=pp
            )[0].squeeze(0)
            img = Image.fromarray(
                np.array(255 * tensr.permute(1, 2, 0)).astype(np.uint8)
            )
            row, col = divmod(j, 5)
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            axes[row, col].set_title(f"True: {classname} \n Predicted {classname}")
            j += 1
    plt.show()


def confused_predictions(model, classname="c0", subject="p026"):
    cam = GradCam(model, device=device)
    df = get_train_df()
    images = df.query("subject == @subject and classname == @classname")["img"]
    preprocessed_images = preprocess_images(images)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    j = 0
    for image in preprocessed_images:
        if j >= 10:
            break
        output = model(image.unsqueeze(0).to(device))
        if output.argmax() != int(classname[-1]):
            tensr = cam(
                input_image=image.unsqueeze(0).to(device), layer=None, postprocessing=pp
            )[0].squeeze(0)
            img = Image.fromarray(
                np.array(255 * tensr.permute(1, 2, 0)).astype(np.uint8)
            )
            row, col = divmod(j, 5)
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            axes[row, col].set_title(
                f"True: {classname} \n Predicted: c{output.argmax()} "
            )
            j += 1
    plt.show()


if __name__ == "__main__":
    model = load_model("aa246d9d2106472492442ff362b1b143")
    confused_predictions(model)
    correct_predictions(model)
