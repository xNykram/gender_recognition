from src.utils.images import load_random_predict_image
from src.train import train_model
import torch
import matplotlib.pyplot as plt   
import numpy as np

model = train_model()

random_predict_image = load_random_predict_image()

model.eval()
with torch.no_grad():
    output = model(random_predict_image.unsqueeze(0))
    _, predicted = torch.max(output.data, 1)

if predicted.item() == 1:
    gender = 'woman'
    print('The model predicts that the person in the image is a woman.')
else:
    gender = 'man'
    print('The model predicts that the person in the image is a man.')

plt.imshow(random_predict_image.squeeze().numpy(), cmap='gray')
plt.title(f'Prediction: {gender}')
plt.savefig('src/output/output.png')