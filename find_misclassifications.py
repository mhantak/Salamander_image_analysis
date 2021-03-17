
from Salamander_lightning_model import SalamanderModel
from salamander_data import getAllImagesDataset
import sys
import torch
import torch.utils.data

if len(sys.argv) != 2:
    exit('ERROR: Please provide the path of a model checkpoint.')

all_images = getAllImagesDataset(
    '/blue/guralnick/mhantak/categories_binary/'
)

dl = torch.utils.data.DataLoader(
    all_images, batch_size=1, shuffle=False, num_workers=1
)

model = SalamanderModel(lr=0.001)
model.load_from_checkpoint(sys.argv[1], lr=0.001)

i = 0
#for img, label in all_images:
for batch, labels in dl:
    print(labels)
    outputs = model(batch)
    #outputs = model(torch.unsqueeze(img, 0))
    #outputs = model(torch.stack([img]))
    p_labels = torch.max(outputs, 1).indices
    print(p_labels, all_images.samples[i])
    i += 1

