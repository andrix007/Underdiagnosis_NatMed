import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPUs 0 and 1

import torch
from classification.train import train
from predictions import make_pred_multilabel
import pandas as pd
from Config import train_df, test_df, val_df


def main():

    MODE = "train"  # Select "train" or "test", "resume"

    if torch.cuda.is_available():
        print("Using GPUs:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available. Running on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if MODE == "train":
        modeltype = "densenet"  
        CRITERION = 'BCELoss'
        lr = 0.5e-3

        model, best_epoch = train( modeltype, CRITERION, device,lr)


    if MODE =="test":
       
        CheckPointData = torch.load('results/checkpoint')
        model = CheckPointData['model']

        make_pred_multilabel(model, device)


    if MODE == "resume":
        modeltype = "resume"  # select 'ResNet50','densenet','ResNet34', 'ResNet18'
        CRITERION = 'BCELoss'
        lr = 0.5e-3

        model, best_epoch = train( modeltype, CRITERION, device,lr)

      


    


if __name__ == "__main__":
    main()
