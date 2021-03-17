
from Salamander_lightning_model import SalamanderModel
from salamander_data import getAllImagesDataset
import sys


if len(sys.argv) != 2):
    exit('ERROR: Please provide the path of a model checkpoint.')

all_images = getAllImagesDataset(
    '/blue/guralnick/mhantak/categories_binary/'
)

model = SalamanderModel()
model.load_from_checkpoint(sys.argv[1])

i = 0
for img, label in all_images:
    outputs = model([img])
    print(outputs, all_images.samples[i])
    i += 1

