
from Salamander_lightning_model import SalamanderModel
from salamander_data import getAllImagesDataset
import sys
import torch
import torch.utils.data
import pandas as pd
import csv 
import numpy as np

if len(sys.argv) != 2:
    exit('ERROR: Please provide the path of a model checkpoint.')

all_images = getAllImagesDataset(
    '/blue/guralnick/mhantak/categories_binary/'
)

dl = torch.utils.data.DataLoader(
    all_images, batch_size=1, shuffle=False, num_workers=1
)

with torch.no_grad():
    model = SalamanderModel.load_from_checkpoint(sys.argv[1], lr=0.001)
    model.eval()
    print(model.feature_extractor._fc.weight)

    i = 0
    correct_cnt = 0
    #for img, labels in all_images:
    for batch, labels in dl:
        print(labels[0])
       # print("all", all_images)
       # print("a_label:", labels)
        outputs = model(batch)
        #outputs = model(torch.unsqueeze(img, 0))
        #outputs = model(torch.stack([img]))
        p_labels = torch.max(outputs, 1).indices
        print("p_label:", p_labels[0], all_images.samples[i])
        #print(p_labels, all_images.samples[i])
        i += 1
        if labels[0] == p_labels[0]:
            correct_cnt += 1
        else:
            print('MISTAKE:')#, all_images.samples[i])
        print(correct_cnt / i, i)


