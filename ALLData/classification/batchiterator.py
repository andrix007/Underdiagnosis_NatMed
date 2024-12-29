import torch
from classification.utils import clip_gradient
import numpy as np
import time


def batch_iterator(model, 
                   phase,
                   dataloader,
                   criterion,
                   optimizer,
                   device):
    '''
        It is a function that iterates over the data in dataloader.
        It evaluates (in val/test phase) or trains (in train phase) the model and returns the loss.
        
        Arguments:
        model : Base model
        phase : A string that denotes if it is "train" or 'val'
        dataloader : The associated data loader
        criterion : Loss function to calculate between predictions and outputs
        optimizer : Optimizer to calculate gradient step from loss
        device: Device on which to run computation
        
    '''
    grad_clip = 0.5
    print_freq = 100  # Print after every 100 batches
    running_loss = 0.0
    batch_count = len(dataloader)

    print(f"{phase.capitalize()} Phase: {batch_count} batches")
    
    # Initialize timers
    epoch_start_time = time.time()
    freq_start_time = time.time()

    for i, data in enumerate(dataloader):
        batch_start_time = time.time()

        imgs, labels, _ = data
        batch_size = imgs.shape[0]
        imgs = imgs.to(device)
        labels = labels.to(device)

        if phase == "train":
            optimizer.zero_grad()
            model.train()
            outputs = model(imgs)
        else:
            model.eval()
            with torch.no_grad():
                outputs = model(imgs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Update weights if in training phase
        if phase == 'train':
            loss.backward()
            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)
            optimizer.step()  # Update weights

        running_loss += loss.item() * batch_size  # Accumulate weighted loss for the batch size

        # Print timing and loss info at intervals or on the last batch
        if (i + 1) % print_freq == 0 or i == batch_count - 1:
            batch_time = time.time() - batch_start_time
            freq_time = time.time() - freq_start_time
            print(f"Batch {i + 1}/{batch_count} | Loss: {loss.item():.4f} | Batch Time: {batch_time:.2f}s")
            print(f"Last {print_freq} Batches Time: {freq_time:.2f}s")
            freq_start_time = time.time()  # Reset frequency timer

    # Epoch timing
    epoch_time = time.time() - epoch_start_time
    print(f"Completed {phase} phase in {epoch_time:.2f} seconds. Total Loss: {running_loss:.4f}")

    return running_loss
