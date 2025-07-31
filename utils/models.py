import torch.nn as nn

class MnistMLP(nn.Module):
    """
    Класс MnistMLP представляет собой полносвязную нейросеть для классификации
    рукописных цифр из датасета MNIST
    """
    def __init__(self) -> None:
        super(MnistMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=784, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MnistCNN(nn.Module):
    """
    Класс MnistCNN представляет собой сверточную нейросеть для классификации
    рукописных цифр из датасета MNIST
    """
    def __init__(self) -> None:
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        return x