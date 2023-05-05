import torch.nn as nn 
import torch
import torchvision.models as models

# This is what Faris told
class CNN(nn.Module):
    """
    CNN classifier. 
    Args:
        num_classes -> number of classes 
        in_features -> features dimension
    Return: logits. 
    """
    def __init__(self, num_classes=2, in_features=196*768):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=768, out_channels=128, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=3*3*64, out_features=128)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        # print(x.shape)
        # x = x.view(-1, 768, 14, 14) # reshape patches to image-like format
        x = x.reshape(-1, 768, 14, 14)
        # print(x.shape)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        # print(x.shape)
        return x
    



