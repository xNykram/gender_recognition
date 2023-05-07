import torchvision.transforms as transforms
import torch
import os
from PIL import Image
from src.core.config import settings
import random

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

def load_traning_images():
    images = []
    labels = []
    for filename in os.listdir(settings.PATH_TRANING_IMAGES):
        if filename.endswith('.jpg'):
            image_path = os.path.join(settings.PATH_TRANING_IMAGES, filename)
            image = Image.open(image_path).resize((128, 128))
            image_transformed = transform(image)
            images.append(image_transformed)
            label = filename.split('_')[0]
            labels.append(label)
    label_to_int = {'male': 0, 'female': 1}
    labels = [label_to_int[label] for label in labels]
    return torch.stack(images), torch.tensor(labels)

def load_random_predict_image():
    image_files = [os.path.join(settings.PATH_PREDICT_IMAGES, f) for f in os.listdir(settings.PATH_PREDICT_IMAGES) if f.endswith('.jpg')]
    random_image = random.choice(image_files)
    image = Image.open(random_image).resize((128, 128))
    print('Processing {}'.format(random_image))
    return transform(image)
