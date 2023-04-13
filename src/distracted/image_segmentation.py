from pathlib import Path
from tqdm import tqdm

import holoviews as hv
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from distracted.data_util import DATA_PATH, MASK_LABELS, C, H, Tensor, W, get_train_df

hv.extension("bokeh")

DISPLAY_IN_NOTEBOOK = False  # Enable to display the panoptic segmentation in notebook


def main():
    df = get_train_df()
    # phone = df[df.desc.str.contains("phone|texting")]
    # good = df[df.desc.str.contains("safe")]
    # imgs = [img() for img in pd.concat([phone.img.iloc[:100], good.img.iloc[:100]])]

    processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-base-coco-panoptic"
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-base-coco-panoptic"
    ).to("cuda")

    onehot_path = DATA_PATH / "onehot"
    onehot_path.mkdir(exist_ok=True)
    for row in tqdm(df.itertuples(), total=len(df)):
        path = (onehot_path / row.path.name).with_suffix(".jpg.npz")
        if path.exists():
            continue
        img = row.img()
        onehot = segmentation_pipeline(img, model, processor)
        save_onehot(path, onehot)


def save_onehot(path: Path, onehot: Tensor[W, H, C]):
    np.savez_compressed(path, data=onehot.to(bool).numpy())


def load_onehot() -> Tensor[W, H, C]:
    return torch.from_numpy(np.load("test.npz")["data"]).to(torch.float32)


def segmentation_pipeline(img: Image.Image, model, processor):
    preprocessed_image = processor(images=img, return_tensors="pt")

    outputs = model(**{k: v.to("cuda") for k, v in preprocessed_image.items()})
    for k, v in list(outputs.items()):
        outputs[k] = v.to("cpu")

    prediction = processor.post_process_panoptic_segmentation(
        outputs,
        target_sizes=[img.size[::-1]],
        label_ids_to_fuse=set(model.config.id2label),
    )[0]

    return extract_onehot(prediction, model.config.id2label)


def extract_onehot(prediction, model_id2label):
    # segment_arr[x, y] == 42 if this pixel belongs to class 42, e.g. "motercycle"
    segment_arr: Tensor[W, H] = prediction["segmentation"]

    # Now we want to normalise the data to only contain classes of interest
    one_hot: Tensor[W, H, C] = torch.zeros(size=segment_arr.shape + (C,))

    for segment in prediction["segments_info"]:
        try:
            class_layer = MASK_LABELS.index(model_id2label[segment["label_id"]])
        except ValueError:
            continue

        one_hot[..., class_layer][segment_arr == segment["id"]] = 1

    if DISPLAY_IN_NOTEBOOK:
        draw_semantic_segmentation(one_hot)

    return one_hot


def draw_semantic_segmentation(one_hot: Tensor[W, H, C]):
    unhot_encode: Tensor[W, H, C] = torch.ones_like(one_hot).cumsum(axis=-1)
    # The following .sum is arbitrary, could also be .max, since each layer is non-overlapping
    class_arr: Tensor[W, H] = (one_hot * unhot_encode).sum(axis=-1)
    num2label = dict(enumerate(["None"] + MASK_LABELS))

    img = hv.Image(class_arr.numpy())
    display(
        img.opts(
            cmap="viridis",
            width=W,
            height=H,
            xaxis=None,
            yaxis=None,
            toolbar=None,
            clim=(-0.5, C + 0.5),
            colorbar=True,
            color_levels=C + 1,
            colorbar_opts={"major_label_overrides": num2label},
        )
    )


if __name__ == "__main__":
    main()
