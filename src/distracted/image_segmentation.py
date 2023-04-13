from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch
from distracted.data_util import get_train_df, Tensor, W, H, MASK_LABELS, C
import pandas as pd
from PIL import Image


def main():
    df = get_train_df()
    phone = df[df.desc.str.contains("phone|texting")]
    good = df[df.desc.str.contains("safe")]
    imgs = [img() for img in pd.concat([phone.img.iloc[:100], good.img.iloc[:100]])]

    processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-base-coco-panoptic"
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-base-coco-panoptic"
    ).to("cuda")

    one_hots: list[Tensor[W, H, C]] = [
        segmentation_pipeline(img, model, processor) for img in imgs[:5]
    ]
    one_hots[-1]
    imgs[4]


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
        if (label := model_id2label[segment["label_id"]]) not in MASK_LABELS:
            continue

        class_layer = MASK_LABELS.index(label)
        one_hot[segment_arr == segment["id"]][..., class_layer] = 1

    return one_hot


def draw_semantic_segmentation(one_hot: Tensor[W, H, C], labels):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import cm

    unhot_encode: Tensor[W, H, C] = torch.ones_like(one_hot).cumsum(axis=-1)
    # The following .sum is arbitrary, could also be .max, since each layer is non-overlapping
    class_arr: Tensor[W, H] = (one_hot * unhot_encode).sum(axis=-1)

    plot_labels = ["None"] + labels
    color_map = cm.get_cmap("viridis", len(plot_labels))

    _, ax = plt.subplots()
    ax.imshow(class_arr, cmap=color_map, vmax=len(plot_labels))

    handles = [
        mpatches.Patch(
            color=color_map(i),
            label=label,
        )
        for i, label in enumerate(plot_labels)
    ]
    ax.legend(handles=handles)
    return ax  # or just show in a notebook


if __name__ == "__main__":
    main()
