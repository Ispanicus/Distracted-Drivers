from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from distracted.data_util import get_train_df
import pandas as pd


processor = AutoImageProcessor.from_pretrained(
    "facebook/mask2former-swin-base-coco-panoptic"
)
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-base-coco-panoptic"
)


def segmentation_pipeline(img):
    preprocessed_image = processor(image, return_tensors="pt")
    outputs = model(**preprocessed_image)

    prediction = processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    return prediction


from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm


def draw_panoptic_segmentation(segmentation, segments_info):
    if segmentation.min() == 1:
        segment_label = segmentation - 1

    class_ids = list({s["label_id"] for s in segments_info})

    color_map = cm.get_cmap("viridis", len(segments_info))

    fig, ax = plt.subplots()
    ax.imshow(segmentation)

    instances_counter = defaultdict(int)
    handles = []
    # for each segment, draw its legend
    for s in segments_info:
        segment_label = model.config.id2label[s["label_id"]]
        label = f"{segment_label}-{instances_counter[s['label_id']]}"
        instances_counter[s["label_id"]] += 1
        color = color_map(s["id"] - 1)
        handles.append(mpatches.Patch(color=color, label=label))

    ax.legend(handles=handles)


df = get_train_df()
phone = df[df.desc.str.contains("phone|texting")]
good = df[df.desc.str.contains("safe")]

df.desc.unique()

imgs = [img() for img in pd.concat([phone.img.iloc[:100], good.img.iloc[:100]])]
model.to("cuda")

results = []
for img in imgs:
    preprocessed_image = processor(images=img, return_tensors="pt")

    outputs = model(**{k: v.to("cuda") for k, v in preprocessed_image.items()})
    for k, v in list(outputs.items()):
        outputs[k] = v.to("cpu")

    predictions = processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[img.size[::-1]]
    )

    for pred in predictions:
        for segment in pred["segments_info"]:
            segment["label"] = model.config.id2label[segment["label_id"]]
    results.append(predictions)

df = pd.DataFrame(results)
infos = [pred[0]["segments_info"] for pred in results]
flattened = [[i, obj] for i, objs in enumerate(infos) for obj in objs]
df = pd.DataFrame(flattened)
df = pd.concat([df, pd.json_normalize(df[1])], axis=1)
df = df.assign(df[0].map(lambda x: "phone" if x < 100 else "good"))
df
pd.json_normalize(flattened)


# Visualise
# for image_func in phone.img.iloc[[100, 200, 300, 400, 500, 600, 700]]:
#     image = image_func()
#     prediction = segmentation_pipeline(image)
#     display(draw_panoptic_segmentation(**prediction))

#     prediction["segments_info"]
#     prediction["segmentation"]
