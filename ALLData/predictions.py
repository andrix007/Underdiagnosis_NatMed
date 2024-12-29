from classification.dataset import AllDatasetsShared
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import sklearn.metrics as sklm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from Config import test_df, val_df


def make_pred_multilabel(model, device):
    """
        This function gives predictions for test fold and calculates AUCs using previously trained model.
        
        Arguments:
        model: densenet-121 from torchvision previously fine tuned to training data
        device: Device on which to run computation
        
        Returns:
        pred_df.csv: dataframe containing individual predictions for each test image
        Threshold.csv: the threshold we used for binary prediction based on maximizing the F1 score over all labels on the validation set
        bipred.csv: dataframe containing individual binary predictions for each test image
        True.csv: dataframe containing true labels
        TestEval.csv: dataframe containing AUCs per label
    """

    BATCH_SIZE = 128
    WORKERS = 12

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    print("Initializing test and validation datasets...")
    dataset_test = AllDatasetsShared(test_df, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize]))
    test_loader = torch.utils.data.DataLoader(dataset_test, BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)

    dataset_val = AllDatasetsShared(val_df, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize]))
    val_loader = torch.utils.data.DataLoader(dataset_val, BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)

    print(f"Test_df size: {len(test_df)}")
    print(f"Val_df size: {len(val_df)}")

    model = model.to(device)
    print(f"Model moved to device: {device}")

    PRED_LABEL = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema']

    for mode in ["Threshold", "test"]:
        print(f"\n--- Starting {mode} mode ---")
        pred_df = pd.DataFrame(columns=["Jointpath"])
        bi_pred_df = pd.DataFrame(columns=["Jointpath"])
        true_df = pd.DataFrame(columns=["Jointpath"])

        if mode == "Threshold":
            loader = val_loader
            Eval_df = pd.DataFrame(columns=["label", 'bestthr'])
            thrs = []
            print("Threshold mode: Using validation set to calculate thresholds.")

        if mode == "test":
            loader = test_loader
            TestEval_df = pd.DataFrame(columns=["label", 'auc', "auprc"])
            print("Test mode: Loading thresholds from file...")
            Eval = pd.read_csv("./results/Threshold.csv")
            thrs = [Eval["bestthr"][Eval[Eval["label"] == label].index[0]] for label in PRED_LABEL]
            print("Loaded thresholds for test mode:", thrs)

        print(f"Processing {len(loader)} batches...")
        for i, data in enumerate(loader):
            print(f"Processing batch {i + 1}/{len(loader)}...")
            inputs, labels, item = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            true_labels = labels.cpu().data.numpy()
            batch_size = true_labels.shape[0]

            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                probs = outputs.cpu().data.numpy()

            # Get predictions and true values for each item in batch
            for j in range(batch_size):
                thisrow = {}
                bi_thisrow = {}
                truerow = {}

                truerow["Jointpath"] = item[j]
                thisrow["Jointpath"] = item[j]
                if mode == "test":
                    bi_thisrow["Jointpath"] = item[j]

                # Iterate over each entry in prediction vector
                for k in range(len(PRED_LABEL)):
                    thisrow["prob_" + PRED_LABEL[k]] = probs[j, k]
                    truerow[PRED_LABEL[k]] = true_labels[j, k]

                    if mode == "test":
                        bi_thisrow["bi_" + PRED_LABEL[k]] = probs[j, k] >= thrs[k]

                pred_df = pd.concat([pred_df, pd.DataFrame([thisrow])], ignore_index=True)
                true_df = pd.concat([true_df, pd.DataFrame([truerow])], ignore_index=True)
                if mode == "test":
                    bi_pred_df = pd.concat([bi_pred_df, pd.DataFrame([bi_thisrow])], ignore_index=True)

            if i % 50 == 0:
                print(f"Processed {i * BATCH_SIZE} samples so far...")

        print(f"Finished processing {mode} batches.")

        # Calculate metrics for each label
        for column in true_df:
            if column not in PRED_LABEL:
                continue
            actual = true_df[column]
            pred = pred_df["prob_" + column]

            thisrow = {"label": column}

            if mode == "test":
                bi_pred = bi_pred_df["bi_" + column]
                thisrow['auc'] = np.nan
                thisrow['auprc'] = np.nan
            else:
                thisrow['bestthr'] = np.nan

            try:
                if mode == "test":
                    thisrow['auc'] = sklm.roc_auc_score(actual.values.astype(int), pred.values)
                    thisrow['auprc'] = sklm.average_precision_score(actual.values.astype(int), pred.values)
                    print(f"Label: {column}, AUC: {thisrow['auc']:.4f}, AUPRC: {thisrow['auprc']:.4f}")
                else:
                    p, r, t = sklm.precision_recall_curve(actual.values.astype(int), pred.values)
                    f1 = 2 * (p * r) / (p + r)
                    bestthr = t[np.argmax(f1)]
                    thrs.append(bestthr)
                    thisrow['bestthr'] = bestthr
                    print(f"Label: {column}, Best Threshold: {bestthr:.4f}")

            except Exception as e:
                print(f"Can't calculate metrics for {column}: {e}")

            if mode == "Threshold":
                Eval_df = pd.concat([Eval_df, pd.DataFrame([thisrow])], ignore_index=True)

            if mode == "test":
                TestEval_df = pd.concat([TestEval_df, pd.DataFrame([thisrow])], ignore_index=True)

        pred_df.to_csv("results/preds.csv", index=False)
        true_df.to_csv("results/True.csv", index=False)

        if mode == "Threshold":
            Eval_df.to_csv("results/Threshold.csv", index=False)

        if mode == "test":
            TestEval_df.to_csv("results/TestEval.csv", index=False)
            bi_pred_df.to_csv("results/bipred.csv", index=False)

    print("AUC avg:", TestEval_df['auc'].mean())
    print("Done!")

    return pred_df, Eval_df, bi_pred_df, TestEval_df
