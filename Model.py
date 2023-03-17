import torchmetrics
import torchvision
import torch

import pytorch_lightning as pl


class Model(pl.LightningModule):
    
    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        num_features = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        #layers[0] = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.feature_extractor = torch.nn.Sequential(*layers)

        # use the pretrained model to classify nothofagus (3 image classes)
        num_target_classes = 3
        self.classifier = torch.nn.Linear(num_features, num_target_classes)        

        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_target_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_target_classes)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.shape[0], -1) # view will flatten the tensor, this is because the Linear layer only accepts a vector (1d array)
        x = self.classifier(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.train_acc(y_hat, y)
        #self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.val_acc(y_hat, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
