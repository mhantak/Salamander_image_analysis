#Salamander color morph machine learning 

import os
from Salamander_lightning_model import SalamanderModel
from salamander_data import getDatasets, getDataLoaders, getAllImagesDataset
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True) 
all_images = getAllImagesDataset('/blue/guralnick/mhantak/categories_binary/')
num_images = len(all_images)
indices = list(range(num_images))

loop_count = 0
for train_idx, valid_idx in kf.split(indices): 
    print('train_idx: %s, valid_idx: %s' % (train_idx, valid_idx)) 
    
    #rng = np.random.default_rng(seed=12345)
    
    train_data, val_data = getDatasets('/blue/guralnick/mhantak/categories_binary/', train_idx=train_idx, valid_idx=valid_idx)
    trainloader, valloader = getDataLoaders(train_data, val_data)

    model = SalamanderModel(lr=float(sys.argv[1]))
    tb_logger = pl_loggers.TensorBoardLogger(
        'logsCrossVal', f'sal_exp_cross_val-{model.lr}'
    )

    checkpoint_callback = ModelCheckpoint(
    dirpath=os.getcwd(),
    save_top_k=1,
    verbose=True,
    monitor='valid_loss',
    mode='min',
    prefix=f'weight-{model.lr}'
    )

    #trainer = pl.Trainer(logger=tb_logger)
    trainer = pl.Trainer(logger=tb_logger, max_epochs=50, gpus=1, checkpoint_callback=checkpoint_callback)

    trainer.fit(model, trainloader, valloader)

    loop_count += 1
    Fold = "fold_"+ str(loop_count)
    print(Fold)