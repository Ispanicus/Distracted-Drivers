import matplotlib.pyplot as plt
import torch
from distracted.dataset_loader import DriverDataset
from transformers import EfficientNetImageProcessor
from torch.utils.data import DataLoader
from PIL import Image
import mlflow
from mirror import mirror
from mirror.visualisations.web import *
#from mirror.visualisations.core import GradCam
from torchvision.transforms import ToTensor, Resize, Compose
from distracted.data_util import DATA_PATH
import torchvision.transforms as T

# create a model
model = mlflow.pytorch.load_model(model_uri='file://D:\Github\Distracted-Drivers\data\mlruns/0/aa246d9d2106472492442ff362b1b143/artifacts/model')
# open some images
# resize the image and make it a tensor
to_input = Compose([Resize((224, 224)), ToTensor()])
images = []
for i in range(10):
    images.append([to_input(Image.open(file)) for file in list(DATA_PATH.glob(f'imgs/train/c{i}/img_*.jpg'))[:2]])

# call mirror with the inputs and the model
mirror(images, model, visualisations=[BackProp, GradCam])


'''cam = GradCam(model, device='cpu')
tensr = cam(to_input(image2).unsqueeze(0), None)[0].squeeze(0)
transform = T.ToPILImage()
img = transform(tensr)
img.show()'''