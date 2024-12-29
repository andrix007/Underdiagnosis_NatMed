import sys
import os

#from Underdiagnosis_NatMed.ALLData.Config import datasets

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPUs 0 and 1
import argparse
import torch
# from classification.train import train
# from predictions import make_pred_multilabel

import pandas as pd
from Config.datasets import train_df, val_df, test_df

from classification_lit.train import LitModel
from classification_lit.data import LitDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from predictions_lit import make_pred_multilabel
from pytorch_lightning import seed_everything

checkpoint_callback = ModelCheckpoint(
    dirpath="results/checkpoint",  # Directory to save checkpoints
    filename="epoch-{epoch:02d}-val_loss-{val_loss:.4f}",  # Naming format
    save_top_k=-1,  # Save all epochs
    verbose=True,
    monitor="val_loss",  # Monitor validation loss
    mode="min",  # Save the checkpoint with the lowest validation loss
    save_weights_only=False  # Save the entire model (including optimizer state)
)


def parse_arguments():
    """
    Parses the command-line arguments to determine the mode of operation.

    Returns:
        str: The mode of operation, which can be 'train', 'test', or 'resume'.

    """

    parser = argparse.ArgumentParser(description="Run the training or testing process.")
    parser.add_argument("--MODE", type=str, required=True, choices=["train", "test", "resume"], 
                        help="Select the mode: 'train', 'test', or 'resume'")
    args = parser.parse_args()
    return args.MODE  # Extract MODE from command-line arguments

def get_device():
    """
    Determines the available computing device (GPU or CPU) and prints the details.
    If GPUs are available, it prints the number of GPUs and their names.
    If no GPUs are available, it prints a message indicating that the code will run on the CPU.
    Returns:
        torch.device: A torch.device object representing the available computing device.
                      It will be 'cuda' if GPUs are available, otherwise 'cpu'.
    """

    if torch.cuda.is_available():
        print("Using GPUs:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available. Running on CPU.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_initial_training_params():
    """
    Retrieves the initial training parameters.
    This function parses the command-line arguments to determine the mode of operation
    and identifies the device (CPU or GPU) to be used for training.
    Returns:
        tuple: A tuple containing:
            - MODE (str): The mode of operation parsed from the command-line arguments.
            - device (torch.device): The device to be used for training.
    """
    
    MODE = parse_arguments()
    device = get_device()

    return (MODE, device)
      
def main_lit():
    
    MODE, device = get_initial_training_params()
    
    if MODE == "train":
        SEED = 3
        seed_everything(SEED, workers=True)

        model = LitModel(model_type="densenet", lr=0.0005, criterion_name="BCELoss", num_labels=8, seed=SEED)

        trainer = Trainer(max_epochs=64, devices=1, accelerator="gpu", callbacks=[checkpoint_callback])

        data_module = LitDataModule(train_df=train_df, val_df=val_df, batch_size=384, num_workers=16)

        trainer.fit(model, datamodule=data_module)
    
    elif MODE == "test":
        model = LitModel.load_from_checkpoint("results/checkpoint/epoch-epoch=01-val_loss-val_loss=0.3116.ckpt")

        data_module = LitDataModule(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,  # Include test dataset
            batch_size=384,
            num_workers=16
        )

        trainer = Trainer(max_epochs=64, devices=1, accelerator="gpu")

        trainer.test(model, datamodule=data_module)

        predictions_file = "test_predictions.pt"
        make_pred_multilabel(predictions_file=predictions_file, results_dir="results")
    
    elif MODE == "resume":
        # Specify the path to the checkpoint file
        checkpoint_path = "lightning_logs/version_3/checkpoints/epoch=0-step=1360.ckpt"  # Update with your checkpoint path

        # Load the model from the checkpoint
        model = LitModel.load_from_checkpoint(checkpoint_path)

        # Initialize the data module
        data_module = LitDataModule(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            batch_size=384,
            num_workers=16
        )

        # Initialize trainer
        trainer = Trainer(
            max_epochs=64,
            devices=1,
            accelerator="gpu",
            callbacks=[checkpoint_callback]
        )

        # Resume training
        trainer.fit(model, datamodule=data_module, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    main_lit()
