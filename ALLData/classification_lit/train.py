import pytorch_lightning as pl
from torchvision import models
from torch import nn
import torch

class LitModel(pl.LightningModule):
    def __init__(self, model_type="densenet", lr=0.0005, criterion_name="BCELoss", num_labels=8):
        super().__init__()
        self.lr = lr
        self.num_labels = num_labels

        # Initialize model
        if model_type == "densenet":
            self.model = models.densenet121(pretrained=True)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(num_ftrs, num_labels),
                nn.Sigmoid()
            )
        elif model_type == "resume":
            checkpoint = torch.load('results/checkpoint')
            self.model = checkpoint['model']

        # Define criterion
        if criterion_name == "BCELoss":
            self.criterion = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        loss = self.criterion(preds, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        loss = self.criterion(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
