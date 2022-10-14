import gradio as gr
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from transformers.models.auto.modeling_auto import \
    AutoModelForImageClassification


def run(image):
    # Preprocess image
    transforms = Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = transforms(image).unsqueeze(0)

    # Pass through model
    prediction = F.softmax(model(pixel_values=image).logits[0], dim=0)
    confidences = {labels[i]: float(prediction[i]) for i in range(len(labels))}

    return confidences


# Load model
ckpt_path = "weights/best-epoch=1-val_acc=0.9526-test_acc=0.9484.ckpt"
ckpt = torch.load(ckpt_path)["state_dict"]

model = AutoModelForImageClassification.from_pretrained(
    "microsoft/beit-base-patch16-224-pt22k-ft22k",
    num_labels=3263,
    ignore_mismatched_sizes=True,
    image_size=224,
)

# Remove prefix from key names
new_state_dict = {}
for k, v in ckpt.items():
    if k.startswith("net"):
        k = k.replace("net" + ".", "")
        new_state_dict[k] = v
model.load_state_dict(new_state_dict, strict=True)

# Load label names
labels = pd.read_csv("classid_classname.csv", names=["id", "name"])["name"].tolist()
labels = [l.replace("_", " ").title() for l in labels]  # Remove _ and capitalize

# Run app
description = """ """

app = gr.Interface(
    title="Classification Model",
    description=description,
    fn=run,
    inputs=[gr.Image(type="pil", tool="select")],
    outputs=gr.Label(num_top_classes=5),
    allow_flagging="never",
)
app.launch()
