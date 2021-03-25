
from Salamander_lightning_model import SalamanderModel
from salamander_data import getAllImagesDataset
import sys
import torch
import torch.utils.data
import torch.nn.functional as F
import pandas as pd
import csv
import re
import numpy as np
from pathlib import Path

if len(sys.argv) != 3:
    exit(
        'ERROR: Please provide a cross-validation output path and an images '
        'folder.\n\nUsage: {0} CV_OUTPUT IMG_FOLDER'.format(sys.argv[0])
    )


# Get the cross-validation model checkpoints.
ckpts = []
cv_dir = Path(sys.argv[1])
for fold_dir in cv_dir.glob('fold_*'):
    if fold_dir.is_dir():
        best_epoch = -1
        best_ckpt = ''
        for ckpt in fold_dir.glob('*.ckpt'):
            m = re.search(r'epoch=([0-9]+)', str(ckpt))
            if m is not None:
                epoch = int(m.group(1))
                if epoch > best_epoch:
                    best_epoch = epoch
                    best_ckpt = str(ckpt)

        ckpts.append(best_ckpt)


all_images = getAllImagesDataset(sys.argv[2])

models = []

with torch.no_grad():
    for i, ckpt in enumerate(ckpts):
        print(f'Loading best model from fold {i}...')
        model = SalamanderModel.load_from_checkpoint(ckpt, lr=0.001)
        model.eval()
        models.append(model)

    i = 0
    correct_cnt = 0
    for img, label in all_images:
        outputs = []
        for model in models:
            output = model(torch.unsqueeze(img, 0))
            output = F.softmax(output, 1)
            outputs.append(torch.squeeze(output))

        outputs = torch.stack(outputs)
        #print(outputs)
        model_avg = torch.mean(outputs, 0)
        #print(model_avg)
        p_label = torch.max(model_avg, 0).indices
        print(label, int(p_label), all_images.samples[i])
        #print(p_labels, all_images.samples[i])
        if label == p_label:
            correct_cnt += 1
        else:
            print('  MISTAKE:', all_images.samples[i])
        i += 1
        print(correct_cnt / i, i)

