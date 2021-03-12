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
for image in all_images: #can also do --- (for co, image in enumerate(all_images):) ---for this don't need the co=0 above of co += 1 below
    print(image)
    color_majority = df[df['file'] == image]['color_majority']
    color_majority = str(list(color_majority)[0]) # Do I need this? Try running without next time 
    if not os.path.exists(os.path.join('categories', color_majority)):
        os.mkdir(os.path.join('categories', color_majority))

    path_from = os.path.join('Sets_1_to_10', image)
    path_to = os.path.join('categories', color_majority, image)

    move(path_from, path_to)
    print('Moved {} to {}'.format(image, path_to))
    co += 1 #can also do (co = co + 1) -- if left co = 0 at the beginning 

print('Moved {} images.'.format(co))
