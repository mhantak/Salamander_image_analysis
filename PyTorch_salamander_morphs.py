#Salamander color morph machine learning 

import os
from Salamander_lightning_model import SalamanderModel
from salamander_data import getDatasets, getDataLoaders
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import sys

rng = np.random.default_rng(seed=12345)

train_data, val_data = getDatasets('/blue/guralnick/mhantak/categories_binary/')
trainloader, valloader = getDataLoaders(train_data, val_data)

model = SalamanderModel(lr=float(sys.argv[1]))
tb_logger = pl_loggers.TensorBoardLogger(
    'logsTEST', f'sal_experiment2-{model.lr}'
)

checkpoint_callback = ModelCheckpoint(
    dirpath=os.getcwd(),
    save_top_k=1,
    verbose=True,
    monitor='valid_loss',
    mode='min',
    prefix=f'weight-{model.lr}'
)

trainer = pl.Trainer(logger=tb_logger, max_epochs=50, gpus=1, checkpoint_callback=checkpoint_callback)
trainer.fit(model, trainloader, valloader)

