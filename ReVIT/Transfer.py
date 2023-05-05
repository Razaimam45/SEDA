import torch.nn as nn 
import torch
import torchvision.models as models

class Classifier(nn.Module): 
    """
    Transfer learning classifier using ResNet-18 pre-trained model. 
    Args:
        num_classes -> number of classes 
    """
    def __init__(self, num_classes=2, freeze=False):
        
        super().__init__()

        # load pre-trained ResNet-18 model
        self.resnet = models.resnet18(pretrained=True)

        # Freeze layers
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # replace last fully connected layer with a new one for classification
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self,x):
        # x = x.reshape(30, 196, 768)
        x = self.resnet(x) #shape is (batch_size, num_features) torch.Size([30, 2])
        return x


# class Classifier(nn.Module): 
#     """
#     Transfer learning classifier using ResNet-18 pre-trained model. 
#     Args:
#         num_classes -> number of classes 
#         num_patches -> number of patches per image
#         hidden_size -> hidden size of each patch feature
#     """
#     def __init__(self, num_classes=2, num_patches=14*14, hidden_size=768, freeze=False):
        
#         super().__init__()
        
#         self.num_patches = num_patches
        
#         # 1D convolutional layer to process patches independently
#         self.conv1d = nn.Conv1d(in_channels=num_patches, out_channels=num_patches, kernel_size=1)

#         # load pre-trained ResNet-18 model
#         self.resnet = models.resnet18(pretrained=True)

#         # Freeze layers
#         if freeze:
#             for param in self.resnet.parameters():
#                 param.requires_grad = False

#         # replace last fully connected layer with a new one for classification
#         num_ftrs = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_ftrs, num_classes)
    
#     def forward(self,x):
#         print(x.shape)
#         x = self.conv1d(x.transpose(1, 2))  # shape is (batch_size, hidden_size, num_patches)
#         x = x.transpose(1, 2)              # shape is (batch_size, num_patches, hidden_size)
#         x = x.reshape(-1, self.num_patches * self.resnet.fc.in_features) # shape is (batch_size, num_patches*512)
#         x = self.resnet(x)                  # shape is (batch_size, num_classes)
#         return x

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=2, freeze=False):
        super(CNN, self).__init__()
        self.freeze = freeze
        self.resnet = models.resnet18(pretrained=True)
        
        # Freeze resnet layers
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Add additional CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 30, kernel_size=3, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )
        
        # replace last fully connected layer with a new one for classification
        self.resnet.fc = nn.Linear(30, num_classes)
        
    def forward(self, x):
        # print(x.shape)
        # x = x.view(-1, 3, 16, 16)
        x = x.reshape(-1, 3, 16, 16)
        # print(x.shape)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        if self.freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
            
        x = self.conv_layers(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        
        return x
    



