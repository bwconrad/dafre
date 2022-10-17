import cv2
import numpy as np
from PIL import Image


def detect(image_pil, cascade_file="app/lbpcascade_animeface.xml"):
    cascade = cv2.CascadeClassifier(cascade_file)

    image = np.array(image_pil)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(24, 24),
    )

    # No faces found
    if len(faces) == 0:
        return image_pil

    x, y, w, h = faces[0]
    face = image[y : y + h, x : x + w]

    return Image.fromarray(face)


if __name__ == "__main__":
    p = "app/detector/3000.jpg"
    img = Image.open(p)
    detect(img)
