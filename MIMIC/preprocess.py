import numpy as np
import pandas as pd

def preprocess_MIMIC(df):

    rename_dict = {
        'Pleural Effusion': 'Effusion',
        'Path': 'path'
    }
    df.rename(columns=rename_dict, inplace=True)

    drop_columns = [
        'Frontal/Lateral', 'AP/PA', 'Enlarged Cardiomediastinum', 'Lung Opacity', 'Lung Lesion', 'Pleural Other', 'Fracture',
        'Support Devices'
    ]

    df.drop(columns=drop_columns, inplace=True, errors='ignore')

    # Step 3: Bin Age using pd.cut() for better control
    df['Age'] = pd.cut(df['Age'], bins=[0, 19, 39, 59, 79, np.inf], labels=["0-20", "20-40", "40-60", "60-80", "80+"])

    # Step 4: Replace specific values in relevant columns
    replacements = {
        'Effusion': {"[False]": 0, "[True]": 1, "[ True]": 1},
        'Gender': {None: 0, -1: 0},
        'Age': {19: "0-20", 39: "20-40", 59: "40-60", 79: "60-80", 81: "80+"}
    }
    df.replace(replacements, inplace=True)
    df['Sex'] = df['Sex'].replace({'Male': 'M', 'Female': 'F'})

    print("Reordering Columns")
    df = df[['subject_id', 'path', 'Sex', 'Age', 'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema']]

    return df


def preprocess_MIMIC(split):
    details = pd.read_csv("/PATH TO MIMIC METADATA/mimic-cxr-metadata-detail.csv")
    details = details.drop(
        columns=['dicom_id', 'study_id', 'religion', 'race', 'insurance', 'marital_status', 'gender'])
    details.drop_duplicates(subset="subject_id", keep="first", inplace=True)
    df = pd.merge(split, details)

    copy_sunbjectid = df['subject_id']
    df.drop(columns=['subject_id'])

    df = df.replace(
        [[None], -1, "[False]", "[True]", "[ True]", 'UNABLE TO OBTAIN', 'UNKNOWN', 'MARRIED', 'LIFE PARTNER',
         'DIVORCED', 'SEPARATED', '0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90',
         '>=90'],
        [0, 0, 0, 1, 1, 0, 0, 'MARRIED/LIFE PARTNER', 'MARRIED/LIFE PARTNER', 'DIVORCED/SEPARATED',
         'DIVORCED/SEPARATED', '0-20', '0-20', '20-40', '20-40', '40-60', '40-60', '60-80', '60-80', '80-', '80-'])

    df['subject_id'] = copy_sunbjectid
    df['Age'] = df["age_decile"]
    df['Sex'] = df["gender"]
    df = df.drop(columns=["age_decile", 'gender'])

    return df