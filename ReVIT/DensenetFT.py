import torch.nn as nn
import torchvision.models as models

class DenseNet121(nn.Module):
    """
    DenseNet121 classifier. 
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
        self.densenet121 = models.densenet121(pretrained=True)
        self.densenet121.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.reshape(-1, 768, 14, 14) # reshape patches to image-like format
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.densenet121.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.densenet121.classifier(x)
        return x
