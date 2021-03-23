#Salamander color morph machine learning

import torch
import numpy as np
import torchvision
from torchvision import transforms


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

# Data augmentation and normalization for training.
train_transform = transforms.Compose([
        transforms.Resize((596,447)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-90, 90)),
        transforms.ColorJitter(hue=.05, saturation=.05),
        transforms.RandomAffine(degrees=0, translate=(.1, .1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

val_transform = transforms.Compose([
        transforms.Resize((596,447)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def getAllImagesDataset(path):
    all_images = torchvision.datasets.ImageFolder(
        root=path, transform=val_transform
    )

    return all_images

def getDatasets(path, train_size=0.75, rng=None, train_idx=None, valid_idx=None):
    if rng is None:
        rng = np.random.default_rng()
   
    trainset = torchvision.datasets.ImageFolder(
        root=path, transform=train_transform
    )
    validset = torchvision.datasets.ImageFolder(
        root=path, transform=val_transform
    )   
   
    if train_idx is None or valid_idx is None:
        num_train = len(trainset)
        indices = list(range(num_train))
        split = int(np.floor(train_size * num_train))
        rng.shuffle(indices)
        train_idx, valid_idx = indices[:split], indices[split:]
        
        traindata = torch.utils.data.Subset(trainset, indices=train_idx)
        valdata = torch.utils.data.Subset(validset, indices=valid_idx)
    else:
        traindata = torch.utils.data.Subset(trainset, indices=train_idx)
        valdata = torch.utils.data.Subset(validset, indices=valid_idx)
    
    return traindata, valdata


def getDataLoaders(
    train_data, val_data, batch_size=8, num_workers=16,
    use_weighted_sampling=True
):
    if use_weighted_sampling:
        # For unbalanced dataset we create a weighted sampler
        weights = make_weights_for_balanced_classes(
            train_data, len(train_data.dataset.classes))
        weights = torch.Tensor(weights)
        sampler1 = torch.utils.data.sampler.WeightedRandomSampler(
            weights, len(weights)
        )
        trainloader = torch.utils.data.DataLoader(
            train_data, sampler=sampler1, batch_size=batch_size, shuffle=False,
            num_workers=num_workers
        )
    else:
        trainloader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=False,
            num_workers=num_workers
        )

    valloader = torch.utils.data.DataLoader(
        val_data, batch_size=8, num_workers=16
    )

    return trainloader, valloader

