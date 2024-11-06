import numpy as np
import pandas as pd

def getPatientID(s: str) -> int:


def preprocess_CXP(df):

    # Step 1: Rename columns if they exist
    rename_dict = {
        'Patient ID': 'subject_id',
        'Patient Age': 'Age',
        'Patient Gender': 'Sex',
        'Image Index': 'path'
    }
    df.rename(columns=rename_dict, inplace=True)

    # Step 2: Drop columns not in universal model (ignoring errors for missing columns)
    drop_columns = [
        'subj_id', 'OriginalImage[Width', 'Finding Labels', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
        'Mass', 'Nodule', 'Pleural Thickening', 'Pneumoperitoneum', 'Pneumomediastinum',
        'Subcutaneous Emphysema', 'Tortuous Aorta', 'Calcification of the Aorta', 'Height]', 'View Position',
        'OriginalImagePixelSpacing[x', 'y]', 'Follow-up #'
    ]
    df.drop(columns=drop_columns, inplace=True, errors='ignore')

    # Step 3: Bin Age using pd.cut() for better control
    df['Age'] = pd.cut(df['Age'], bins=[0, 19, 39, 59, 79, np.inf], labels=["0-20", "20-40", "40-60", "60-80", "80+"])

    # Step 4: Replace specific values in relevant columns
    replacements = {
        'Pleural Effusion': {"[False]": 0, "[True]": 1, "[ True]": 1},
        'Gender': {None: 0, -1: 0},
        'Age': {19: "0-20", 39: "20-40", 59: "40-60", 79: "60-80", 81: "80+"}
    }
    df.replace(replacements, inplace=True)

    print("Reordering Columns")
    df = df[['subject_id', 'path', 'Sex', 'Age', 'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema']]

    return df


def preprocess_CXP(split):
    split['Age'] = np.where(split['Age'].between(0, 19), 19, split['Age'])
    split['Age'] = np.where(split['Age'].between(20, 39), 39, split['Age'])
    split['Age'] = np.where(split['Age'].between(40, 59), 59, split['Age'])
    split['Age'] = np.where(split['Age'].between(60, 79), 79, split['Age'])
    split['Age'] = np.where(split['Age'] >= 80, 81, split['Age'])

    copy_sunbjectid = split['subject_id']
    split.drop(columns=['subject_id'])
    split = split.replace([[None], -1, "[False]", "[True]", "[ True]", 19, 39, 59, 79, 81],
                          [0, 0, 0, 1, 1, "0-20", "20-40", "40-60", "60-80", "80-"])

    split['subject_id'] = copy_sunbjectid
    split['Sex'] = np.where(split['Sex'] == 'Female', 'F', split['Sex'])
    split['Sex'] = np.where(split['Sex'] == 'Male', 'M', split['Sex'])

    return split