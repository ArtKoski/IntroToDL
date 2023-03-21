import os
import numpy as np
from PIL import Image

def load_images(image_folder):
    image_files = os.listdir(image_folder)
    images = [Image.open(os.path.join(image_folder, img_file)) for img_file in image_files]
    return images

def calculate_mean_std(images):
    num_channels = 3  # for RGB images
    total_images = len(images)

    mean = np.zeros(num_channels)
    std = np.zeros(num_channels)

    for img in images:

        if img.mode != 'RGB':
                img = img.convert("RGB")

        img_array = np.array(img) / 255.0  # Normalize pixel values to the range [0, 1]
        for channel in range(num_channels):
            mean[channel] += img_array[:, :, channel].mean()
            std[channel] += img_array[:, :, channel].std()

    mean /= total_images
    std /= total_images

    return mean, std

image_folder = "/home/vixmaria/koulu/IDL/projekti/IntroToDL-main/dataset/images"
images = load_images(image_folder)
mean, std = calculate_mean_std(images)

print(f"Mean: {mean}")
print(f"Standard deviation: {std}")