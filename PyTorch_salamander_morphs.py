#Salamander color morph machine learning 

import torch, os
#os.chdir("/Users/maggie/Dropbox/P.cinereus_ML/Salamander_ML_code/")
from Salamander_lightning_model import SalamanderModel
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.metrics import ConfusionMatrix
from efficientnet_pytorch import EfficientNet
from pytorch_lightning.callbacks import ModelCheckpoint
import sys

rng = np.random.default_rng(seed=12345)

####Oversampling
def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses   
    labels = []                                                   
    for cnt, item in enumerate(images):                                                         
        count[item[1]] += 1   
        labels.append(item[1])
        print(cnt)                                                  
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(labels):                                          
        weight[idx] = weight_per_class[val]                                  
    return weight 

# Data augmentation and normalization for training
transform = transforms.Compose([
        transforms.Resize((596,447)), 
        transforms.RandomHorizontalFlip(p=0.5),  
        transforms.RandomVerticalFlip(p=0.5),  
        transforms.RandomRotation(degrees=(-90, 90)), 
        transforms.ColorJitter(hue=.05, saturation=.05),
        transforms.RandomAffine(degrees=0, translate=(.1, .1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

transformNoAugment = transforms.Compose([
        transforms.Resize((596,447)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#Loading datasets 
trainset = torchvision.datasets.ImageFolder(root='/blue/guralnick/mhantak/categories_binary/', transform=transform)
validset = torchvision.datasets.ImageFolder(root='/blue/guralnick/mhantak/categories_binary/', transform=transformNoAugment)

# Create the index splits for training and validation
train_size = 0.75
num_train = len(trainset)
indices = list(range(num_train))
split = int(np.floor(train_size * num_train))
#np.random.shuffle(indices)
rng.shuffle(indices)
train_idx, valid_idx = indices[:split], indices[split:] 
print(len(train_idx)) #~3000
print(len(valid_idx)) #~1000

traindata = torch.utils.data.Subset(trainset, indices=train_idx)
valdata = torch.utils.data.Subset(validset, indices=valid_idx)

# For unbalanced dataset we create a weighted sampler                       
weights = make_weights_for_balanced_classes(traindata, len(trainset.classes))                                                             
weights = torch.Tensor(weights)                                    
sampler1 = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)) 

trainloader = torch.utils.data.DataLoader(traindata, sampler=sampler1, batch_size=8, shuffle = False, num_workers=16) 
valloader = torch.utils.data.DataLoader(valdata, batch_size=8, num_workers=16) 

if __name__ == "__main__":  

    model = SalamanderModel(lr=float(sys.argv[1]))
    tb_logger = pl_loggers.TensorBoardLogger('logsTEST', f'sal_experiment2-{model.lr}')

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd(),
        save_top_k=1,
        verbose=True,
        monitor='valid_loss',
        mode='min',
        prefix=f'weight-{model.lr}'
    )
    
    trainer = pl.Trainer(logger=tb_logger, max_epochs=50, gpus=1, checkpoint_callback=checkpoint_callback)
    trainer.fit(model, trainloader, valloader)