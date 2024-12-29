import time
import csv
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import datetime
import torch.optim
import torch.utils.data
from torchvision import  models
from torch import nn
import torch
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from .dataset import AllDatasetsShared
from classification.utils import  checkpoint, save_checkpoint, saved_items
from classification.batchiterator import batch_iterator
from tqdm import tqdm
import random
import numpy as np
from Config import train_df, val_df


def train(modeltype, CRITERION, device, lr):
    # Training parameters
    BATCH_SIZE = 384
    WORKERS = 16
    N_LABELS = 8
    start_epoch = 0
    num_epochs = 64
    #num_epochs = 64

    val_df_size = len(val_df)
    print(f"Validation dataset size: {val_df_size}")
   
    train_df_size = len(train_df)
    print(f"Training dataset size: {train_df_size}")

    random_seed = 6  # Set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Normalize transformations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        AllDatasetsShared(train_df, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        AllDatasetsShared(val_df, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True
    )

    # Model setup
    if modeltype == 'densenet':
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    elif modeltype == 'resume':
        CheckPointData = torch.load('results/checkpoint')
        model = CheckPointData['model']

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Criterion setup
    if CRITERION == 'BCELoss':
        criterion = nn.BCELoss().to(device)

    # Initialize tracking
    epoch_losses_train = []
    epoch_losses_val = []
    since = time.time()

    best_loss = float('inf')
    best_epoch = -1

    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # Start training loop
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Training Progress"):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Learning rate: {lr}")
        print('-' * 40)
        epoch_start = time.time()

        # Training phase
        phase = 'train'
        running_loss = batch_iterator(model, phase, train_loader, criterion, optimizer, device)
        epoch_loss_train = running_loss / train_df_size
        epoch_losses_train.append(epoch_loss_train)
        print(f"Training Loss: {epoch_loss_train:.4f}")

        # Validation phase
        phase = 'val'
        running_loss = batch_iterator(model, phase, val_loader, criterion, optimizer, device)
        epoch_loss_val = running_loss / val_df_size
        epoch_losses_val.append(epoch_loss_val)
        print(f"Validation Loss: {epoch_loss_val:.4f}")

        # Checkpoint if best validation loss
        if epoch_loss_val < best_loss:
            best_loss = epoch_loss_val
            best_epoch = epoch
            checkpoint(model, best_loss, best_epoch, lr)
            print(f"New best model saved with Validation Loss: {best_loss:.4f}")

        # Log training and validation loss
        with open("results/log_train", 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            if epoch == 0:
                logwriter.writerow(["epoch", "train_loss", "val_loss", "seed", "lr"])
            logwriter.writerow([epoch, epoch_loss_train, epoch_loss_val, random_seed, lr])

        # Epoch timing
        epoch_duration = time.time() - epoch_start
        print(f"Epoch completed in {epoch_duration // 60:.0f}m {epoch_duration % 60:.0f}s")

        # Early stopping or learning rate adjustment
        if (epoch - best_epoch) >= 3:
            print(f"No improvement in validation loss for 3 epochs. Reducing learning rate from {lr} to {lr / 2}.")
            lr /= 2
            if (epoch - best_epoch) >= 5:
                print("No improvement for 5 epochs. Stopping early.")
                break

    # Total training time
    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    saved_items(epoch_losses_train, epoch_losses_val, time_elapsed, BATCH_SIZE)

    # Load best model checkpoint
    checkpoint_best = torch.load('results/checkpoint')
    model = checkpoint_best['model']
    best_epoch = checkpoint_best['best_epoch']
    print(f"Best model from epoch: {best_epoch}")

    return model, best_epoch