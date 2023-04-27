from transformers.models.efficientnet.modeling_efficientnet import (
    EfficientNetForImageClassification,
)

EfficientNetForImageClassification.__call__ = lambda self, x: self.forward(x).logits

if __name__ == "__main__":
    from distracted.experimental_setups import PREPROCESSOR
    from distracted.data_util import get_train_df, DATA_PATH
    from distracted.gradcam import correct_predictions, confused_predictions
    import mlflow
    import numpy as np
    from PIL import Image
    import torch

    df = get_train_df()
    for idx, row in df.iterrows():
        break
    img = row.img()
    pre_img = PREPROCESSOR(img)["pixel_values"][0]
