# Distracted-Drivers

# Goal
Fine-tune and apply feature engineering and adapters to Efficientnet for a classification task and examine the classwise accuracy consequences.
## Fine tuning
This simply corresponds to retrainig the top layers at a high learning rate, and the body layers at a lower learning rate, such that we maintain the core of the model knowledge, but quickly converge towards the ends to our task specific classifications. This is implemented by a few lines in the main file [classifier.py](src\distracted\classifiers.py)

## Feature engineering: [Image segmentation](.\src\distracted\segmentation_nn.py)
Big picture: At layer L-1, we not only feed it the outputs from L-2, but feed it a larger vector

Old L-1:
  (classifier): Linear(in_features=1536, out_features=10)

New L-1
  (classifier): Linear(in_features=1536 + len(feature_engineering), out_features=10)

The feature engineering we apply is 6 classes from a semantic segmentation (Technically it is a panoptic segmentation, merged to become semantic). The classes are
```
    person
    bottle
    cell phone
    cup
    car
    chair
```
See an [image example](cellphone-example.png). These are passed through a simple CNN and flattened to create the feature_engineering vector

## [Adapters](.\src\distracted\adapters.py)
By placing a layer adjacent to a low-level layer, we can fine-tune low-level features to our task, while only having to store the much smaller adapter layer.

# Training
```
python -m distracted.classifiers [OPTIONS]

Options:
  --batch-size INTEGER
  --model-name TEXT
  --adapters TEXT
  --top-lr FLOAT
  --top-decay FLOAT
  --body-lr FLOAT
  --body-decay FLOAT
  --gamma FLOAT
  --epochs INTEGER
  --adapter-weight FLOAT
  --checkpoint TEXT
```
Which model is trained? See [classifiers.py](src\distracted\classifiers.py):
```
    if adapters:
        setup = adapter_setup(...)
    elif body_lr:
        setup = finetune_setup(...)
    else:
        setup = segmentation_setup(...)
```
Models and metrics are stored via [MLFlow.org](https://mlflow.org/):
```
cd ./data && mlflow ui
```
Final evaluations are performed via [gradcam.py](src\distracted\gradcam.py). It is recommended running this file using [VSCode interactive](https://code.visualstudio.com/docs/python/jupyter-support-py) for an interactive data exploration experience.

# Hardware requirements
A batch size of 128 is optimal for 12 GB of VRAM. It is highly recommended to have at least 16GB of RAM, preferably 32GB.

# Setup
```
git clone https://github.com/Ispanicus/Distracted-Drivers
cd Distracted-Drivers
./install_windows.bat
```
Not tested on mac or linux, but the install script should be similar if not identical.

Download the "distracted driver" dataset
https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data

Unzip and rename to Distracted-Drivers/data/

Then cache panoptic masks (Duration: Hours)
```
python -m distracted.image_segmentation.py
```
