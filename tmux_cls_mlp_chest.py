import functions
import predictions
from predictions import Prediction
from features import Features
import mlp
from mlp import Classifier, ClassifierCLS
import data
import attacks
import torch
import pandas as pd
import numpy as np
import torch.nn as nn 
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns 
import random
import pickle
import torch 
from sklearn.metrics import auc
from autoattack import AutoAttack
from torchvision.utils import save_image
import math
from matplotlib.backends.backend_pdf import PdfPages
import train_model
import os
from ensemble_whotebox import New
import timm 
import torchsummary
import sys
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torchsummary import summary
import pickle

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']= '4'

epochs = 100
lr = 1e-3

vit = '/home/faris.almalik/Desktop/Thesis /Models/Models_(Pre_trained)VIT/vit_base_patch16_224_in21k_test-accuracy_0.96.pth'
image_size = (224,224)
batch_size = 30

model = torch.load(vit).cuda()

for w in model.parameters(): 
    w.requires_grad = False
data_loader, image_dataset = data.data_loader(batch_size= batch_size, image_size=image_size)


test_list = {f'block_{i}': [] for i in range(12)}
epochs = 60
lr = 1e-3
for index in range(12): 
    classifier = ClassifierCLS(num_classes=2, in_features=768).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(),lr = lr)
    scheduler = StepLR(optimizer=optimizer, step_size=15, gamma=0.1, verbose=True)

    classifier.train() 
    for epoch in range(epochs): 
        train_acc = 0.0
        train_loss = 0.0 
        print(f'Epoch {epoch+1}/{epochs}')
        for image, label in tqdm(data_loader['train']):
            image = image.cuda()
            label = label.cuda()
            x = model.patch_embed(image)
            x = torch.cat((model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = model.pos_drop(x + model.pos_embed)
            for i in range(index+1):
                x = model.blocks[i](x)
            output = classifier(x[:,0,:])
            prediction = torch.argmax(output , dim= -1)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            acc = sum(prediction == label).float().item()/len(label)
            train_acc += acc 
            
        scheduler.step() 
        train_acc = train_acc/len(data_loader['train'])   
        train_loss = train_loss/len(data_loader['train'])
        print(f'train_acc= {train_acc:.2f}')
        print (f'train_loss = {train_loss:.2f}')    
    
    #Test loop 
    print('Testing')
    classifier.eval()
    test_acc = 0.0
    test_loss = 0.0

    with torch.no_grad(): 
        for image, label in tqdm(data_loader['test']):
            image = image.cuda()
            label = label.cuda()
            x = model.patch_embed(image)
            x = torch.cat((model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = model.pos_drop(x + model.pos_embed)
            for i in range(index+1):
                x = model.blocks[i](x)
            output = classifier(x[:,0,:])
            loss = criterion(output, label)
            acc = sum(torch.argmax(output, dim=-1) == label).float().item()/len(label)
            test_acc += acc
            test_loss += loss.item()
    test_acc = test_acc/len(data_loader['test'])
    test_loss = test_loss/len(data_loader['test'])
    print(f'test_acc = {test_acc:.2f}')
    print (f'test_loss = {test_loss:.2f}')
    test_list[f'block_{index}'].append(test_acc)
    print(f'================= Block {index} Finished ==============')

with open("accuracy_list_chest_mlp_cls_latest", "wb") as fp:   #Pickling
     pickle.dump(test_list, fp)