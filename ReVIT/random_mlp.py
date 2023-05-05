import torch.nn as nn 
import torch

# class Classifier(nn.Module): 
#     """
#     MLP classifier. 
#     Args:
#         num_classes -> number of classes 
#         in_feature -> features dimension

#     return logits. 
    
#     """
#     def __init__(self,num_classes=2 ,in_features = 768*196):
        
#         super().__init__()
#         self.linear1 = nn.Linear(in_features= in_features, out_features= 4096)
#         self.linear2 = nn.Linear(in_features= 4096, out_features= 2048)
#         self.linear3 = nn.Linear(in_features= 2048, out_features= 128)
#         self.linear4 = nn.Linear(in_features= 128, out_features= num_classes)
#         self.dropout = nn.Dropout(0.3)
    
#     def forward(self,x):
#         x= x.reshape(-1, 196*768)
#         # print(x.shape) # torch.Size([30, 150528])
#         x = nn.functional.relu(self.linear1(x))
#         x = nn.functional.relu(self.linear2(x))
#         x = nn.functional.relu(self.linear3(x))
#         x = self.linear4(x)
#         return x


class Classifier(nn.Module):
    """
    MLP classifier.
    Args:
        num_classes -> number of classes
        in_feature -> features dimension
        hidden_dim -> hidden dimension of each linear layer
        num_layers -> number of layers

    return logits.

    """
    def __init__(self, num_classes=2, in_features=768*196, hidden_dim=4096, num_layers=4):

        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features=in_features, out_features=hidden_dim))

        for i in range(num_layers - 2):
            hidden_dim = hidden_dim // 2  # Reduce hidden dim by factor of 2
            self.layers.append(nn.Linear(hidden_dim*2, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, num_classes))
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.reshape(-1, 196*768)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = nn.functional.relu(x)
                x = self.dropout(x)
        return x

import torch

class AdversarialClassifier(nn.Module):
    """
    MLP classifier.
    Args:
        num_classes -> number of classes
        in_features -> features dimension

    return logits.

    """
    def __init__(self, num_classes=2, in_features=768*196):

        super().__init__()
        self.linear1 = nn.Linear(in_features=in_features, out_features=4096)
        self.linear2 = nn.Linear(in_features=4096, out_features=2048)
        self.linear3 = nn.Linear(in_features=2048, out_features=128)
        self.linear4 = nn.Linear(in_features=128, out_features=num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.reshape(-1, 196*768)
        x = self.fgsm_attack(x, epsilon=0.03)
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        x = nn.functional.relu(self.linear3(x))
        x = self.linear4(x)
        return x

    def fgsm_attack(self, x, epsilon):
        """
        Fast Gradient Sign Method (FGSM) attack.
        Args:
            x: input tensor
            epsilon: perturbation magnitude

        return adversarial example
        """
        x_adv = x.detach().cpu()
        x_adv.requires_grad = True
        logits = self.forward(x_adv)
        pred = logits.argmax(dim=1)
        loss = nn.CrossEntropyLoss()(logits, pred)
        self.zero_grad()
        loss.backward()

        # Generate perturbation
        x_grad = x_adv.grad.data.sign()
        x_advers = x_adv + epsilon * x_grad

        return x_advers

