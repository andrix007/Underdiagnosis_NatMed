import torch
from torch.utils.data import Dataset
import os
import numpy as np
from imageio import imread
from PIL import Image


class AllDatasetsShared(Dataset):
    def __init__(self, dataframe, finding="any", transform=None):
        """
        Dataset class representing the aggregation of all three datasets.
        Initially, in the Config.py, we have aggregated all CheXpert, MIMIC-CXR, and ChestX-ray14 datasets on the 8 shared labels.

        Arguments:
        dataframe: Whether the dataset represents the train, test, or validation split
        PATH_TO_IMAGES: Path to the image directory on the server
        transform: Whether to conduct transforms on the images or not

        Returns:
        image, label, and item["Jointpath"] as the unique indicator of each item in the dataloader.
        """
        self.dataframe = dataframe
        self.dataset_size = self.dataframe.shape[0]
        self.transform = transform
        self.PRED_LABEL = [
            'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion',
            'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema'
        ]
        AllDatasetsShared.missing_files = []
        

    def __getitem__(self, idx):
        while True:  # Loop until a valid file is found
            item = self.dataframe.iloc[idx]

            try:
                # Read the image
                img = imread(item["Jointpath"])

                # Ensure the image has 3 channels
                if len(img.shape) == 2:  # Grayscale image
                    img = np.stack([img] * 3, axis=-1)  # Convert to RGB
                elif img.shape[2] > 3:  # More than 3 channels
                    img = img[:, :, :3]  # Use the first 3 channels

                # Ensure correct dtype
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)

                # Convert to PIL image
                img = Image.fromarray(img)

                # Apply transformations if provided
                if self.transform is not None:
                    img = self.transform(img)

                # Generate labels
                label = torch.zeros(len(self.PRED_LABEL), dtype=torch.float32)
                for i, pred_label in enumerate(self.PRED_LABEL):
                    value = self.dataframe[pred_label.strip()].iloc[idx]
                    if not np.isnan(value):  # Skip NaN values
                        label[i] = value.astype('float')

                return img, label, item["Jointpath"]

            except FileNotFoundError:
                #AllDatasetsShared.missing_files.append(item["Jointpath"])
                #print(f"File not found: {item['Jointpath']}")
                with open('missing_files.txt', 'a') as mf:
                    mf.write(item['Jointpath']+'\n')
                idx = (idx + 1) % self.dataset_size  # Move to the next index

    def __len__(self):
        return self.dataset_size
