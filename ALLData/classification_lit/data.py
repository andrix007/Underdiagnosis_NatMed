import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from .dataset import AllDatasetsShared  # Assuming this is your Dataset class
from Config.datasets import train_df, val_df, test_df

class LitDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df=None, batch_size=384, num_workers=16):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Normalization values for ImageNet (if using pre-trained models)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def setup(self, stage=None):
        # Define transformations
        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            self.normalize,
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            self.normalize,
        ])

        # Assign datasets
        self.train_dataset = AllDatasetsShared(self.train_df, transform=self.train_transforms)
        self.val_dataset = AllDatasetsShared(self.val_df, transform=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)
    
    def test_dataloader(self):
        if self.test_df is None:
            raise ValueError("Test dataset (test_df) is not provided.")
        
        return DataLoader(
            AllDatasetsShared(self.test_df, transform=self.val_transforms),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
