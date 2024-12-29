import pandas as pd
import pytorch_lightning as pl
from torchvision.models import densenet121, DenseNet121_Weights
from torch import nn
import torch

class LitModel(pl.LightningModule):
    def __init__(self, model_type="densenet", lr=0.0005, criterion_name="BCELoss", num_labels=8, seed=-1):
        super().__init__()
        self.lr = lr
        self.num_labels = num_labels
        self.seed = seed
        self.epoch_logs = []  # List to store epoch logs
        self.current_train_loss = None  # Initialize to track train loss
        self.current_lr = lr  # Initialize learning rate

        # Initialize model
        if model_type == "densenet":
            self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
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
        imgs, labels, _ = batch
        preds = self(imgs)
        loss = self.criterion(preds, labels)

        k = 50  # Log every 'k' batches
        if batch_idx % k == 0:
            current_lr = self.optimizers().param_groups[0]["lr"]
            gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GPU memory in GB
            print(f"Batch {batch_idx}: Loss = {loss.item():.4f}, LR = {current_lr:.6f}, GPU Mem = {gpu_memory:.2f} GB")

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.current_train_loss = loss  # Store the current training loss

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels, _ = batch
        preds = self(imgs)
        loss = self.criterion(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        imgs, labels, paths = batch
        preds = self(imgs)
        loss = self.criterion(preds, labels)

        # Save outputs for further processing
        self.test_outputs.append({"preds": preds, "labels": labels, "paths": paths})

        # Log the test loss
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"preds": preds, "labels": labels, "paths": paths}

    def on_train_epoch_end(self):
        """Update learning rate and track train loss at the end of the epoch."""
        optimizer = self.trainer.optimizers[0]
        self.current_lr = optimizer.param_groups[0]["lr"]  # Update current learning rate

    def on_validation_epoch_end(self):
        """Log validation metrics and save them."""
        val_loss = self.trainer.callback_metrics.get("val_loss", None)

        # Log epoch information
        epoch_log = {
            "epoch": self.current_epoch,
            "train_loss": self.current_train_loss.item() if self.current_train_loss is not None else "N/A",
            "val_loss": val_loss.item() if val_loss is not None else "N/A",
            "seed": self.seed,
            "lr": self.current_lr if hasattr(self, "current_lr") else "N/A",
        }
        self.epoch_logs.append(epoch_log)

        # Print log
        print(
            f"Epoch {self.current_epoch}: "
            f"Train Loss = {epoch_log['train_loss']}, "
            f"Val Loss = {epoch_log['val_loss']}, "
            f"Seed = {epoch_log['seed']}, "
            f"LR = {epoch_log['lr']}"
        )

        # Save logs to a CSV file
        pd.DataFrame(self.epoch_logs).to_csv("epoch_logs.csv", index=False)


    def on_test_epoch_start(self):
        self.test_outputs = []  # Initialize a list to store outputs

    def on_test_epoch_end(self):
        all_preds = torch.cat([x["preds"] for x in self.test_outputs], dim=0)
        all_labels = torch.cat([x["labels"] for x in self.test_outputs], dim=0)
        all_paths = sum([x["paths"] for x in self.test_outputs], [])  # Concatenate paths

        # Save predictions for analysis
        print("Saving test predictions...")
        torch.save({"preds": all_preds, "labels": all_labels, "paths": all_paths}, "test_predictions.pt")

        # Clear test outputs to free memory
        self.test_outputs = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
