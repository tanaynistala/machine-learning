################################################################################################
import torch
import numpy as np
from torch.nn import Sequential
from torch.nn import Conv2d, MaxPool2d, Dropout2d, Flatten, Linear, ReLU, BatchNorm2d, Sequential, AvgPool1d, Softmax, LazyLinear

# NOTE: you CANNOT import anything else from torch or other deep learning packages. It
# means you need to construct a CNN using these layers. If you need other layers, you can ask us
# first, but you CANNOT use existing models such as ResNet from tensorflow.
################################################################################################

class GlobalAvgPool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, x):
        return torch.mean(x, dim=[2, 3])

def ConvNet(**kwargs):
    """
    Construct a CNN using `torch`: https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html.
    """

    # TODO: implement your own model
    layers = [
        Conv2d(3, 16, [7, 7], stride=3),
        ReLU(),
        BatchNorm2d(16),
        Conv2d(16, 32, [5, 5], stride=2),
        ReLU(),
        BatchNorm2d(32),
        Conv2d(32, 64, [3, 3], stride=1),
        ReLU(),
        BatchNorm2d(64),
        Conv2d(64, 128, [1, 1], stride=1),
        ReLU(),
        BatchNorm2d(128),
        Conv2d(128, 256, [1, 1], stride=1),
        GlobalAvgPool2d(),
        LazyLinear(3)
    ]

    model = Sequential(*layers)

    return model


def train(model, train_loader, valid_loader, learning_rate, max_epoch, device):

    # Move the model to device
    model.to(device)

    # Set up variables to recode losses and accuracies later
    train_loss = []
    val_loss   = []
    val_acc    = []

    # TODO: set up the optimizer and the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # TODO: implement the training procedure. Please remember to zero out previous gradients in each iteration.
    for i in range(max_epoch):
        for batch, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # TODO: within the training loop, please calculate and record the validation loss and accuracy
            if batch % 100 == 0:
                val_loss.append(0)
                val_acc.append(0)
                for batch, (images, labels) in enumerate(valid_loader):
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss[-1] += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_acc[-1] += (predicted == labels).sum().item()

                val_loss[-1] /= len(valid_loader)
                val_acc[-1] /= len(valid_loader)

        train_loss.append(0)
        for batch, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss[-1] += loss.item()

        train_loss[-1] /= len(train_loader)

        # If you want, you can uncomment this line to print losses
        print(f"Epoch [{i+1}]: Training Loss: {train_loss[-1]} Validation Loss: {val_loss[-1]} Accuracy: {val_acc[-1]}")


    return train_loss, val_loss, val_acc
