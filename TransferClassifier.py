import torch.nn as nn
import torchvision.models as models

class TransferLearningModel(nn.Module):
    """
    Transfer learning model based on ResNet18.

    Args:
        num_classes (int): number of output classes.
        hidden_size (int): hidden size of the ViT model.
    """
    def __init__(self, num_classes=2, hidden_size=768):
        super().__init__()

        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Identity()  # remove the original fully connected layer

        # Create new layers
        self.conv1 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.resnet18.layer1
        self.conv3 = self.resnet18.layer2
        self.conv4 = self.resnet18.layer3
        self.conv5 = self.resnet18.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Reshape input tensor from shape [batch_size, num_patches, hidden_size] to shape [batch_size, hidden_size, sqrt(num_patches), sqrt(num_patches)]
        batch_size, num_patches, hidden_size = x.shape
        patch_size = int(hidden_size ** 0.5)
        x = x.view(-1, 196, 768)
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(batch_size, patch_size, patch_size * num_patches)
        print((x.shape))
        # Pass through convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Pass through fully connected layers
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x
