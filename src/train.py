import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.images import load_traning_images
from torch import device, cuda
from src.models.gender_classifier import GenderClassifier
from src.core.config import settings

def train_model():
    device('gpu') if cuda.is_available() else device('cpu')
    images, labels = load_traning_images()
    dataset = torch.utils.data.TensorDataset(images, labels)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    model = GenderClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = settings.NUM_EPOCH
    print(type(num_epochs))
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataset)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))
    return model
