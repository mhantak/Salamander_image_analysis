#Salamander model
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet 
import torch.nn.functional as F
from pytorch_lightning.metrics import ConfusionMatrix

##LIGHTNING
class SalamanderModel(pl.LightningModule):

    def __init__(self, lr):
        super().__init__()
        # init a pretrained transfer learning model
        num_target_classes = 2 
        self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_target_classes)
        self.lr = lr
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.valid_conf = pl.metrics.ConfusionMatrix(num_classes=2) # was 5

    def forward(self, x):
        representations = self.feature_extractor(x)
        return representations

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        #print(outputs, labels)
        p_labels = torch.max(outputs, 1).indices
        print(p_labels, labels)
        loss = F.cross_entropy(outputs, labels)
        accuracy = self.train_acc(outputs, labels)
        print(accuracy)
        self.log_dict({'train_loss': loss, 'train_acc': accuracy}, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        #print(outputs, labels)
        p_labels = torch.max(outputs, 1).indices
        print("valid_labels:", p_labels, labels)
        loss = F.cross_entropy(outputs, labels)
        accuracy = self.valid_acc(outputs, labels)
        print(accuracy)
        self.valid_conf.update(p_labels, labels)
        self.log_dict({'valid_loss': loss, 'vaild_acc': accuracy}, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        print(self.valid_conf.compute())

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9) 
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=8, verbose=True, factor=0.1)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, 
           'monitor': 'valid_loss'
       }
        
        #return [optimizer], [scheduler]
        #return optimizer
