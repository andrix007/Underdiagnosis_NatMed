import numpy as np
import pandas as pd

def getPatientID(s: str) -> int:
    #CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg
    patient_string = s.split('/')[2]
    nr = ""
    for ch in patient_string:
        if ch.isdigit():
            nr += ch
    return int(nr)

def preprocess_CXP(df):

    df.insert(1, 'subject_id', df['Path'].apply(getPatientID))

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
