from src.utils.images import load_random_predict_image
from src.train import train_model
import torch

model = train_model()

random_predict_image = load_random_predict_image()

model.eval()
with torch.no_grad():
    output = model(random_predict_image.unsqueeze(0))
    _, predicted = torch.max(output.data, 1)

if predicted.item() == 1:
    print('The model predicts that the person in the image is a woman.')
else:
    print('The model predicts that the person in the image is a man.')
