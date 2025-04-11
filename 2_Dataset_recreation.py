import os
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("final_cleaned_file.csv")
print("Original data shape:", df.shape)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_ratio = 0.60  # 60% for training
dev_ratio = 0.25    # 25% for development
test_ratio = 0.15   # 15% for testing

train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=42)
print("Training set shape:", train_df.shape)

dev_frac = dev_ratio / (dev_ratio + test_ratio)
dev_df, test_df = train_test_split(temp_df, test_size=(1 - dev_frac), random_state=42)
print("Development set shape:", dev_df.shape)
print("Test set shape:", test_df.shape)

output_dir = "Final_data"
os.makedirs(output_dir, exist_ok=True)

train_df.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
dev_df.to_csv(os.path.join(output_dir, "dev_data.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)

print("Data splits saved in 'Final_data' folder as train_data.csv, dev_data.csv, and test_data.csv")