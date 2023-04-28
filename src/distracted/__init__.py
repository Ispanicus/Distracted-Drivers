from pathlib import Path

from transformers.models.efficientnet.modeling_efficientnet import (
    EfficientNetForImageClassification,
)

import distracted

EfficientNetForImageClassification.__call__ = lambda self, x: self.forward(x).logits
pre_commit = Path(distracted.__path__[0]).parents[1] / ".git/hooks/pre-commit"
if not pre_commit.exists():
    raise IOError(
        "Please run `pip install pre-commit && pre-commit install` to enable pre-commit hooks"
    )

if __name__ == "__main__":
    import mlflow
    import numpy as np
    import torch
    from PIL import Image

    from distracted.data_util import DATA_PATH, get_train_df
    from distracted.experimental_setups import PREPROCESSOR
    from distracted.gradcam import confused_predictions, correct_predictions

    df = get_train_df()
    for idx, row in df.iterrows():
        break
    img = row.img()
    pre_img = PREPROCESSOR(img)["pixel_values"][0]
