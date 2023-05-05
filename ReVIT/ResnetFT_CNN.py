import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    """
    ResNet18 classifier. 
    Args:
        num_classes -> number of classes 
        in_features -> features dimension
    Return: logits. 
    """
    def __init__(self, num_classes=2, in_features=196*768):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=768, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resnet18 = models.resnet18(pretrained=True)
        self.additional_convs = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.resnet18.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.reshape(-1, 768, 14, 14) # reshape patches to image-like format
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.additional_convs(x) # pass through additional conv layers
        x = self.resnet18.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.resnet18.fc(x)
        return x