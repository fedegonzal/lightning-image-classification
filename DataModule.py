from pathlib import Path
import pytorch_lightning as pl

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class DataModule(pl.LightningDataModule):
    
    def __init__(self, path="./", batch_size=64):
        self.batch_size = batch_size
        self.num_workers = 2
        self.path = Path(path)
        self.prepare_data_per_node = True
        self._log_hyperparams = True
        self.allow_zero_length_dataloader_with_multiple_devices = True
    
    #def prepare_data(self): # download, tokenize ...
    #    return
        
    #def setup(self): # split, transforms, normalize ...
    #    return
    
    def train_dataloader(self):
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        dataset = datasets.ImageFolder(self.path / "train/", train_transforms)
        loader = DataLoader(dataset, 
            batch_size=self.batch_size, shuffle=True, 
            num_workers=self.num_workers, pin_memory=True
        )
        return loader


    def val_dataloader(self):
        val_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        dataset = datasets.ImageFolder(self.path / "validation/", val_transforms)
        loader = DataLoader(dataset, 
            batch_size=self.batch_size, shuffle=True, 
            num_workers=self.num_workers, pin_memory=True
        )
        return loader
