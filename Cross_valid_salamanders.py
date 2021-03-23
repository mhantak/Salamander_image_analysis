#Salamander color morph machine learning 

import os
from Salamander_lightning_model import SalamanderModel
from salamander_data import getDatasets, getDataLoaders, getAllImagesDataset
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
import os.path
import pathlib
from sklearn.model_selection import KFold


if len(sys.argv) != 3:
    exit(
        '\nERROR: Please provide a learning rate and output folder name.\n\n'
        'Usage: {0} LEARNING_RATE OUTPUT_FOLDER\n'.format(sys.argv[0])
    )

learning_rate = float(sys.argv[1])
run_folder = sys.argv[2]

if os.path.isdir(run_folder):
    exit(f'\nThe output folder {run_folder} already exists.\n')

outpath = pathlib.Path(run_folder)
outpath.mkdir()

kf = KFold(n_splits=4, shuffle=True) 
all_images = getAllImagesDataset('/blue/guralnick/mhantak/categories_binary/')
num_images = len(all_images)
indices = list(range(num_images))

loop_count = 0
for train_idx, valid_idx in kf.split(indices): 
    print('train_idx: %s, valid_idx: %s' % (train_idx, valid_idx))

    fold_folder = outpath / "fold_"+ str(loop_count)
    fold_folder.mkdir()
    print(f'Cross-validation fold {loop_count}; saving results to {fold_folder}.')
    
    #rng = np.random.default_rng(seed=12345)
    
    train_data, val_data = getDatasets(
        '/blue/guralnick/mhantak/categories_binary/',
        train_idx=train_idx, valid_idx=valid_idx
    )
    trainloader, valloader = getDataLoaders(train_data, val_data)

    model = SalamanderModel(lr=learning_rate)
    tb_logger = pl_loggers.TensorBoardLogger(
        str(fold_folder), f'sal_exp_cross_val-{model.lr}'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(fold_folder),
        save_top_k=1,
        verbose=True,
        monitor='valid_loss',
        mode='min',
        prefix=f'weight-{model.lr}'
    )

    #trainer = pl.Trainer(logger=tb_logger)
    trainer = pl.Trainer(
        logger=tb_logger, max_epochs=50, gpus=1, checkpoint_callback=checkpoint_callback
    )

    trainer.fit(model, trainloader, valloader)

    loop_count += 1

