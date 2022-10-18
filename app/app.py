import gradio as gr
import pandas as pd
import torch
import torch.nn.functional as F
from detect import detect
from huggingface_hub import hf_hub_download
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from transformers.models.auto.modeling_auto import \
    AutoModelForImageClassification


def run(image, auto_crop):
    if auto_crop:
        image = detect(image, "app/lbpcascade_animeface.xml")

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

    # Denormalize image
    image.clamp_(min=float(image.min()), max=float(image.max()))
    image.add_(-float(image.min())).div_(float(image.max()) - float(image.min()) + 1e-5)
    image = image.squeeze(0).permute(1, 2, 0).numpy()

    return confidences, image


# Load model
ckpt_path = hf_hub_download(
    "bwconrad/beit-base-patch16-224-pt22k-ft22k-dafre",
    "beit-base-patch16-224-pt22k-ft22k-dafre.ckpt",
)
ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))["state_dict"]

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
labels = pd.read_csv("app/classid_classname.csv", names=["id", "name"])["name"].tolist()
labels = [l.replace("_", " ").title() for l in labels]  # Remove _ and capitalize

# Run app
description = """
A character classification model trained on the DAF:re dataset which consists of 3263 characters from anime, manga and video game series. 
A list of all characters can be found [here](https://github.com/bwconrad/dafre/blob/main/app/classid_classname.csv).

The model is trained and performs best on head and shoulder portrait images. 
Users can manually crop images through the UI or check the `auto_crop` box to let a face detection model do the cropping.
"""

app = gr.Interface(
    title="Anime Character Classification",
    description=description,
    fn=run,
    inputs=[gr.Image(type="pil", tool="select"), gr.Checkbox(label="auto_crop")],
    outputs=[gr.Label(num_top_classes=5), gr.Image().style(height=224, width=224)],
    allow_flagging="never",
    examples=[
        ["app/examples/rei.jpg"],
        ["app/examples/futaba.jpg"],
        ["app/examples/yotsuba.jpg"],
    ],
)
app.launch()
