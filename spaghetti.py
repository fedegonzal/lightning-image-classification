import pytorch_lightning as pl
#from pytorch_lightning.callbacks.early_stopping import EarlyStopping

#from torch.utils import data

from DataModule import DataModule
from Model import Model


if __name__ == '__main__':
    #freeze_support()

    data = DataModule(path="nothofagus")
    model = Model()

    trainer = pl.Trainer(accelerator='mps', devices=-1, max_epochs=5, precision=16, callbacks=[
        # EarlyStopping(monitor='val_acc', mode='max', verbose=True)
    ])
    trainer.fit(model, data)
