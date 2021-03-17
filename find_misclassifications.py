
from Salamander_lightning_model import SalamanderModel
from salamander_data import getAllImagesDataset
import sys
import torch

if len(sys.argv) != 2:
    exit('ERROR: Please provide the path of a model checkpoint.')

all_images = getAllImagesDataset(
    '/blue/guralnick/mhantak/categories_binary/'
)

model = SalamanderModel(lr=0.001)
model.load_from_checkpoint(sys.argv[1], lr=0.001)

i = 0
for img, label in all_images:
    print(label)
    outputs = model(torch.unsqueeze(img, 0))
    #outputs = model(torch.stack([img]))
    p_labels = torch.max(outputs, 1).indices
    print(p_labels, all_images.samples[i])
    i += 1

