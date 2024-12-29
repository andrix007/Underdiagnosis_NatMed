import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

file_path_train = os.path.join(BASE_DIR, "../data_mevis/ALLData/preprocessed_train_df_1.csv")
file_path_test = os.path.join(BASE_DIR, "../data_mevis/ALLData/preprocessed_test_df_1.csv")
file_path_val = os.path.join(BASE_DIR, "../data_mevis/ALLData/preprocessed_val_df_1.csv")

test_df = pd.read_csv(file_path_test)
train_df = pd.read_csv(file_path_train)
val_df = pd.read_csv(file_path_val)