import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Paths to your original files
train_file = "./preprocessed_train_df_1.csv"  # Path to train.csv
val_file = "./preprocessed_val_df_1.csv"      # Path to val.csv
test_file = "./preprocessed_test_df_1.csv"    # Path to test.csv

output_folder = "./split_hospitals"  # Output folder for hospital datasets
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the datasets
print("Loading original datasets...")
train_df = pd.read_csv(train_file)
val_df = pd.read_csv(val_file)
test_df = pd.read_csv(test_file)

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

# Step 1: Split train.csv into 3 parts
train_h1, train_temp = train_test_split(train_df, test_size=0.67, random_state=42)
train_h2, train_h3 = train_test_split(train_temp, test_size=0.5, random_state=42)

# Step 2: Split val.csv into 3 parts
val_h1, val_temp = train_test_split(val_df, test_size=0.67, random_state=42)
val_h2, val_h3 = train_test_split(val_temp, test_size=0.5, random_state=42)

# Step 3: Split test.csv into 3 parts
test_h1, test_temp = train_test_split(test_df, test_size=0.67, random_state=42)
test_h2, test_h3 = train_test_split(test_temp, test_size=0.5, random_state=42)

# Step 4: Save the splits
print("Saving hospital datasets...")

# Hospital 1
train_h1.to_csv(f"{output_folder}/hospital_1_train.csv", index=False)
val_h1.to_csv(f"{output_folder}/hospital_1_val.csv", index=False)
test_h1.to_csv(f"{output_folder}/hospital_1_test.csv", index=False)

# Hospital 2
train_h2.to_csv(f"{output_folder}/hospital_2_train.csv", index=False)
val_h2.to_csv(f"{output_folder}/hospital_2_val.csv", index=False)
test_h2.to_csv(f"{output_folder}/hospital_2_test.csv", index=False)

# Hospital 3
train_h3.to_csv(f"{output_folder}/hospital_3_train.csv", index=False)
val_h3.to_csv(f"{output_folder}/hospital_3_val.csv", index=False)
test_h3.to_csv(f"{output_folder}/hospital_3_test.csv", index=False)

# Print completion message
print("Datasets have been split and saved successfully!")
print(f"Files are saved in: {output_folder}")
