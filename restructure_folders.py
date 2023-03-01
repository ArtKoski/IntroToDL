import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Set the paths for the image and annotation directories
image_dir = 'images'
annotation_dir = 'annotations'

# Define the names of the classes
classes = ['baby', 'bird', 'car', 'clouds', 
           'dog', 'female', 'flower', 'male', 
           'night', 'people', 'portrait', 'river', 
           'sea', 'tree']

# Set the ratio of the split
train_ratio = 0.8
dev_ratio = 0.1
test_ratio = 0.1

# Get the list of image filenames
image_filenames = os.listdir(image_dir)
image_filenames = [f for f in image_filenames if f.endswith('.jpg')]

# Split the image filenames into train, dev, and test sets
train_val_filenames, test_filenames = train_test_split(image_filenames, test_size=test_ratio, random_state=42)
train_filenames, dev_filenames = train_test_split(train_val_filenames, test_size=dev_ratio/(train_ratio+dev_ratio), random_state=42)

# Create the train, dev, and test subdirectories
for subfolder in ['train', 'dev', 'test']:
    os.makedirs(os.path.join('dataset', subfolder), exist_ok=True)
    for c in classes:
        os.makedirs(os.path.join('dataset', subfolder, c), exist_ok=True)

# Iterate over each text file in the annotations directory
for file_name in os.listdir(annotation_dir):
    if not file_name.endswith('.txt'):
        continue

    # Read the list of image indexes from the text file
    with open(os.path.join(annotation_dir, file_name)) as f:
        indexes = [int(line.strip()) for line in f]

    # Create lists of train, dev, and test image indexes for this class
    train_indexes = [index for index in indexes if f"im{index}.jpg" in train_filenames]
    dev_indexes = [index for index in indexes if f"im{index}.jpg" in dev_filenames]
    test_indexes = [index for index in indexes if f"im{index}.jpg" in test_filenames]

    # Copy the corresponding images to the appropriate subdirectory
    for index, subfolder in zip(train_indexes, ['train']*len(train_indexes)):
        image_name = f"im{index}.jpg"
        source_path = os.path.join(image_dir, image_name)
        target_path = os.path.join('dataset', subfolder, file_name.split('.')[0], image_name)
        shutil.copy(source_path, target_path)

    for index, subfolder in zip(dev_indexes, ['dev']*len(dev_indexes)):
        image_name = f"im{index}.jpg"
        source_path = os.path.join(image_dir, image_name)
        target_path = os.path.join('dataset', subfolder, file_name.split('.')[0], image_name)
        shutil.copy(source_path, target_path)

    for index, subfolder in zip(test_indexes, ['test']*len(test_indexes)):
        image_name = f"im{index}.jpg"
        source_path = os.path.join(image_dir, image_name)
        target_path = os.path.join('dataset', subfolder, file_name.split('.')[0], image_name)
        shutil.copy(source_path, target_path)

