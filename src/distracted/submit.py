import torch
import torchvision.io
from tqdm import tqdm

from distracted.data_util import (
    DATA_PATH,
    C,
    H,
    W,
    get_train_df,
    load_model,
    load_onehot,
    timeit,
)
from distracted.experimental_setups import PREPROCESSOR

header = ["img", "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]
result = [header]
model = load_model("23057eb022344d8eb097a6a8ce1522eb").to("cuda")
for path in tqdm(list(DATA_PATH.glob("imgs/test/*.jpg"))):
    img = torchvision.io.read_image(str(path))
    pre = PREPROCESSOR(img)["pixel_values"][0]
    tensor = torch.from_numpy(pre).to("cuda").unsqueeze(0)
    softmax = model(tensor)[0].detach().cpu()
    result.append([path.name] + softmax.tolist())

import pandas as pd

df = pd.DataFrame(result[1:], columns=result[0])
df
output = "\n".join(",".join(str(v) for v in result))
with open("submission.csv", "w") as f:
    f.write(output)
