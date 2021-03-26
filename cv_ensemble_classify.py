
from Salamander_lightning_model import SalamanderModel
from salamander_data import getAllImagesDataset
import torch
import torch.utils.data
import torch.nn.functional as F
import csv
import re
from pathlib import Path
import os.path
from argparse import ArgumentParser


argp = ArgumentParser(
    description='Uses a cross-validation ensemble to analyze images.'
)
argp.add_argument(
    '-c', '--cv_dir', type=str, required=True, 
    help='The path to a cross-validation output directory.'
)
argp.add_argument(
    '-i', '--images', type=str, required=True,
    help='The path to a collection of images.'
)
argp.add_argument(
    '-o', '--output', type=str, required=False, default='',
    help='The path of an output CSV file.'
)
argp.add_argument(
    '-a', '--accuracy', action='store_true',
    help='If set, record image set classifications and track accuracy.'
)

args = argp.parse_args()

# Get the cross-validation model checkpoints.
ckpts = []
cv_dir = Path(args.cv_dir)
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


all_images = getAllImagesDataset(args.images)

if args.output != '':
    writer = csv.DictWriter(
        open(args.output, 'w'),
        ['file', 'prediction', '0', '1']
    )
    writer.writeheader()
else:
    writer = None

rowout = {}
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
        imgfile = all_images.samples[i][0]

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
        print(label, int(p_label), imgfile)
        #print(p_labels, all_images.samples[i])

        if writer is not None:
            rowout['file'] = os.path.basename(imgfile)
            rowout['prediction'] = int(p_label)
            rowout['0'] = float(model_avg[0])
            rowout['1'] = float(model_avg[1])
            writer.writerow(rowout)

        if args.accuracy:
            if label == p_label:
                correct_cnt += 1
            else:
                print('  MISTAKE:', all_images.samples[i])
        i += 1
        if args.accuracy:
            print('Cumulative accuracy:', correct_cnt / i, i)

print(i, 'total images processed.')

