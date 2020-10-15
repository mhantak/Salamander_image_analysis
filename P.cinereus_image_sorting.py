#Read in .csv file with all salamander scores

import pandas as pd
from shutil import move
import os

os.chdir('/Users/maggie/Dropbox/P.cinereus_ML/Consensus_scores_NAs_updated/')

df = pd.read_csv('All_salamander_scores.csv')
#df = pd.read_csv('/Users/maggie/Dropbox/P.cinereus_ML/Consensus_scores_NAs_updated/All_salamander_scores.csv')
print(df.head())

#Need to sort images into seperate folders based on image labels (e.g. R, L, U, O)

all_images = os.listdir('Sets_1_to_10')

co = 0
for image in all_images:
    print(image)
    color_majority = df[df['file'] == image]['color_majority']
    color_majority = str(list(color_majority)[0])
    if not os.path.exists(os.path.join('categories', color_majority)):
        os.mkdir(os.path.join('categories', color_majority))

    path_from = os.path.join('Sets_1_to_10', image)
    path_to = os.path.join('categories', color_majority, image)

    move(path_from, path_to)
    print('Moved {} to {}'.format(image, path_to))
    co += 1 

print('Moved {} images.'.format(co))
