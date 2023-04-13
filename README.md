# Distracted-Drivers

# Goal
Fine-tune and apply feature engineering to Efficientnet for a classification task

Specifically, at layer L-1, we not only feed it the outputs from L-2, but feed it a larger vector

L-2: 
  (classifier): Linear(in_features=2560, out_features=1000, bias=True)

New L-2
  (classifier): Linear(in_features=2560 + len(feature_engineering), out_features=1000, bias=True)

# Requirements
CUDA & Pytorch

# Setup
```
git clone https://github.com/Ispanicus/Distracted-Drivers
pip install -e Distracted-Drivers
```

Download the "distracted driver" dataset
https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data

Unzip and rename to Distracted-Drivers/data/

```
echo Caching panoptic masks
python -m image_segmentation.py
```


