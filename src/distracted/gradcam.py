from mirror.visualisations.core import GradCam
import matplotlib.pyplot as plt
import torch
from PIL import Image
import mlflow
from distracted.data_util import get_train_df
from transformers import EfficientNetImageProcessor
import numpy as np

MODEL_NAME = "google/efficientnet-b3"
PREPROCESSOR = EfficientNetImageProcessor.from_pretrained(
    MODEL_NAME
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#default model
#model = mlflow.pytorch.load_model(model_uri='file://D:\Github\Distracted-Drivers\data\mlruns/0/aa246d9d2106472492442ff362b1b143/artifacts/model')

def pp(x):
    x = (x-x.min())
    x = (x/x.max())
    return (x).clamp(0, 1)

def preprocess_images(images):
    preprocessed_images = []
    for image_func in images:
        preprocessed_images.append(PREPROCESSOR(image_func(), return_tensors="pt")['pixel_values'].squeeze())
    return preprocessed_images

def correct_predictions(model, classname = 'c0', subject = 'p026'):
    type(model).__call__ = lambda self, x: self.forward(x).logits
    cam = GradCam(model, device=device)
    df = get_train_df()
    images = df.query("subject == @subject and classname == @classname")['img']
    preprocessed_images = preprocess_images(images)    

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    j = 0
    for image in preprocessed_images:
        if j >= 10:
            break
        output = model(image.unsqueeze(0).to(device))
        if output.argmax() == int(classname[-1]):
            tensr = cam(input_image = image.unsqueeze(0).to(device), layer=None, postprocessing=pp)[0].squeeze(0)
            img = Image.fromarray(np.array(255*tensr.permute(1,2,0)).astype(np.uint8))
            row, col = divmod(j, 5)
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            axes[row, col].set_title(f'True: {classname} \n Predicted {classname}')
            j += 1
    plt.show()

def confused_predictions(model, classname = 'c0', subject = 'p026'):
    type(model).__call__ = lambda self, x: self.forward(x).logits
    cam = GradCam(model, device=device)
    df = get_train_df()
    images = df.query("subject == @subject and classname == @classname")['img']
    preprocessed_images = preprocess_images(images)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    j = 0
    for image in preprocessed_images:
        if j >= 10:
            break
        output = model(image.unsqueeze(0).to(device))
        if output.argmax() != int(classname[-1]):
            tensr = cam(input_image = image.unsqueeze(0).to(device), layer=None, postprocessing=pp)[0].squeeze(0)
            img = Image.fromarray(np.array(255*tensr.permute(1,2,0)).astype(np.uint8))
            row, col = divmod(j, 5)
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            axes[row, col].set_title(f'True: {classname} \n Predicted: c{output.logits.argmax()} ')
            j += 1
    plt.show()