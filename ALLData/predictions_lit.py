import torch
import pandas as pd
import numpy as np
import sklearn.metrics as sklm
from Config.datasets import test_df, val_df


def make_pred_multilabel(predictions_file="test_predictions.pt", results_dir="results"):
    """
    Process saved predictions and calculate metrics like AUC and AUPRC.

    Arguments:
    predictions_file: Path to the .pt file containing test predictions, labels, and paths.
    results_dir: Directory to save the evaluation results.

    Outputs:
    Saves predictions, thresholds, binary predictions, true labels, and evaluation metrics to CSV files.
    """
    # Load predictions from the saved .pt file
    data = torch.load(predictions_file, weights_only=True)  # Secure loading
    preds = data["preds"].cpu().numpy()
    labels = data["labels"].cpu().numpy()
    paths = data["paths"]

    # Prediction labels
    PRED_LABEL = [
        "No Finding", "Atelectasis", "Cardiomegaly", "Pleural Effusion",
        "Pneumonia", "Pneumothorax", "Consolidation", "Edema"
    ]

    # DataFrames for results
    pred_df = pd.DataFrame({"Jointpath": paths})
    true_df = pd.DataFrame({"Jointpath": paths})
    bi_pred_df = pd.DataFrame({"Jointpath": paths})
    TestEval_df = pd.DataFrame(columns=["label", "auc", "auprc"])
    Eval_df = pd.DataFrame(columns=["label", "bestthr"])
    thresholds = []

    # Populate DataFrames
    for i, label in enumerate(PRED_LABEL):
        pred_df[f"prob_{label}"] = preds[:, i]
        true_df[label] = labels[:, i]

    # Load thresholds or calculate them if missing
    try:
        threshold_file = f"{results_dir}/Threshold.csv"
        Eval = pd.read_csv(threshold_file)
        thresholds = [Eval.loc[Eval["label"] == label, "bestthr"].values[0] for label in PRED_LABEL]
        print(f"Loaded thresholds: {thresholds}")
    except FileNotFoundError:
        print("Threshold file not found. Calculating thresholds from validation set...")
        thresholds = []
        for i, label in enumerate(PRED_LABEL):
            p, r, t = sklm.precision_recall_curve(labels[:, i], preds[:, i])
            f1 = 2 * (p * r) / (p + r + 1e-8)
            best_thr = t[np.argmax(f1)]
            thresholds.append(best_thr)
            Eval_df = pd.concat([Eval_df, pd.DataFrame([{"label": label, "bestthr": best_thr}])], ignore_index=True)

        Eval_df.to_csv(f"{results_dir}/Threshold.csv", index=False)

    # Calculate binary predictions and metrics
    for i, label in enumerate(PRED_LABEL):
        bi_pred_df[f"bi_{label}"] = preds[:, i] >= thresholds[i]

        try:
            auc = sklm.roc_auc_score(labels[:, i], preds[:, i])
            auprc = sklm.average_precision_score(labels[:, i], preds[:, i])
            TestEval_df = pd.concat([TestEval_df, pd.DataFrame([{"label": label, "auc": auc, "auprc": auprc}])], ignore_index=True)
            print(f"{label}: AUC = {auc:.4f}, AUPRC = {auprc:.4f}")
        except Exception as e:
            print(f"Error calculating metrics for {label}: {e}")

    # Save results
    pred_df.to_csv(f"{results_dir}/preds.csv", index=False)
    true_df.to_csv(f"{results_dir}/True.csv", index=False)
    bi_pred_df.to_csv(f"{results_dir}/bipred.csv", index=False)
    TestEval_df.to_csv(f"{results_dir}/TestEval.csv", index=False)

    print(f"AUC avg: {TestEval_df['auc'].mean():.4f}")
    print("Results saved to:", results_dir)

    return pred_df, Eval_df, bi_pred_df, TestEval_df
