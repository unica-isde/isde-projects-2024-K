import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from app.config import Configuration

def generate_histogram(image_id):
    conf = Configuration()
    image_path = os.path.join(conf.image_folder_path, image_id)
    output_dir = os.path.join("app", "static", "generated")
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    np_img = np.array(img)

    plt.figure()
    for i, color in enumerate(['r', 'g', 'b']):
        hist = np.histogram(np_img[..., i], bins=256, range=(0, 256))[0]
        plt.plot(hist, color=color)

    plt.title(f"Histogram for {image_id}")
    output_path = os.path.join(output_dir, f"{image_id}_hist.png")
    plt.savefig(output_path)
    plt.close()

    return f"{image_id}_hist.png"