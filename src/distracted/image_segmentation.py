from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from distracted.data_util import get_train_df


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
df.desc.unique()

for image_func in phone.img.iloc[[100, 200, 300, 400, 500, 600, 700]]:
    image = image_func()
    prediction = segmentation_pipeline(image)
    display(draw_panoptic_segmentation(**prediction))

    prediction["segments_info"]
    prediction["segmentation"]
