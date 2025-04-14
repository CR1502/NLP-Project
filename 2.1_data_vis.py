############################################################################
"""
This file Just has extra visualizations for the presentation and report
"""
############################################################################

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

train_path = os.path.join("Final_data", "train_data.csv")
dev_path = os.path.join("Final_data", "dev_data.csv")
test_path = os.path.join("Final_data", "test_data.csv")

print("Loading processed data...")
train_df = pd.read_csv(train_path)
dev_df = pd.read_csv(dev_path)
test_df = pd.read_csv(test_path)

print("Train DataFrame shape:", train_df.shape)
print("Dev DataFrame shape:", dev_df.shape)
print("Test DataFrame shape:", test_df.shape)

def analyze_class_distribution(df, dataset_name):
    class_counts = df["label"].value_counts()
    class_perc = df["label"].value_counts(normalize=True) * 100
    print(f"\nClass Distribution for {dataset_name}:")
    print(class_counts)
    print(f"\nClass Percentage for {dataset_name} (%):")
    print(class_perc)
    class_counts.to_csv(f"{dataset_name}_class_distribution.csv")
    plt.figure(figsize=(6,4))
    sns.countplot(x="label", data=df, order=class_counts.index, palette="viridis")
    plt.title(f"{dataset_name} - Class Distribution")
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"Images/{dataset_name}_class_distribution.png", dpi=300)
    plt.close()

analyze_class_distribution(train_df, "train_data")
analyze_class_distribution(dev_df, "dev_data")
analyze_class_distribution(test_df, "test_data")


def compute_length(text):
    return len(text.split())

train_df["claim_length"] = train_df["claim"].apply(compute_length)
dev_df["claim_length"] = dev_df["claim"].apply(compute_length)
test_df["claim_length"] = test_df["claim"].apply(compute_length)

train_length_summary = train_df["claim_length"].describe()
dev_length_summary = dev_df["claim_length"].describe()
test_length_summary = test_df["claim_length"].describe()

print("\nTrain Claim Length Summary:")
print(train_length_summary)
print("\nDev Claim Length Summary:")
print(dev_length_summary)
print("\nTest Claim Length Summary:")
print(test_length_summary)

train_length_summary.to_csv("CSV files/train_claim_length_summary.csv")
dev_length_summary.to_csv("CSV files/dev_claim_length_summary.csv")
test_length_summary.to_csv("CSV files/test_claim_length_summary.csv")

# Plot histograms for claim lengths
plt.figure(figsize=(10, 4))
plt.subplot(1,3,1)
sns.histplot(train_df["claim_length"], bins=30, kde=True, color="skyblue")
plt.title("Train Claim Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")

plt.subplot(1,3,2)
sns.histplot(dev_df["claim_length"], bins=30, kde=True, color="salmon")
plt.title("Dev Claim Length Distribution")
plt.xlabel("Number of Words")

plt.subplot(1,3,3)
sns.histplot(test_df["claim_length"], bins=30, kde=True, color="lightgreen")
plt.title("Test Claim Length Distribution")
plt.xlabel("Number of Words")

plt.tight_layout()
plt.savefig("Images/claim_length_distribution_all.png", dpi=300)
plt.close()

train_df["dataset"] = "Train"
dev_df["dataset"] = "Dev"
test_df["dataset"] = "Test"
combined_df = pd.concat([train_df, dev_df, test_df], axis=0)

plt.figure(figsize=(10,6))
sns.countplot(x="label", hue="dataset", data=combined_df,
              palette="viridis", order=combined_df["label"].value_counts().index)
plt.title("Class Distribution Across Train, Dev, and Test Sets")
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("Images/combined_class_distribution.png", dpi=300)
plt.close()

print("Visualizations and summaries saved successfully!")


############################################################################
# This is the result that I got after running this file
"""
Loading processed data...
Train DataFrame shape: (5338, 6)
Dev DataFrame shape: (2224, 6)
Test DataFrame shape: (1335, 6)

Class Distribution for train_data:
label
1    3332
0    2006
Name: count, dtype: int64

Class Percentage for train_data (%):
label
1    62.420382
0    37.579618
Name: proportion, dtype: float64
/Users/craigroberts/Documents/Coding/NLP/NLP-Project/2.1_data_vis.py:42: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.countplot(x="label", data=df, order=class_counts.index, palette="viridis")

Class Distribution for dev_data:
label
1    1404
0     820
Name: count, dtype: int64

Class Percentage for dev_data (%):
label
1    63.129496
0    36.870504
Name: proportion, dtype: float64
/Users/craigroberts/Documents/Coding/NLP/NLP-Project/2.1_data_vis.py:42: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.countplot(x="label", data=df, order=class_counts.index, palette="viridis")

Class Distribution for test_data:
label
1    820
0    515
Name: count, dtype: int64

Class Percentage for test_data (%):
label
1    61.423221
0    38.576779
Name: proportion, dtype: float64
/Users/craigroberts/Documents/Coding/NLP/NLP-Project/2.1_data_vis.py:42: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.countplot(x="label", data=df, order=class_counts.index, palette="viridis")

Train Claim Length Summary:
count    5338.000000
mean       13.308543
std         7.609113
min         3.000000
25%         8.000000
50%        10.000000
75%        16.000000
max        82.000000
Name: claim_length, dtype: float64

Dev Claim Length Summary:
count    2224.000000
mean       13.220773
std         7.108522
min         4.000000
25%         9.000000
50%        11.000000
75%        16.000000
max        60.000000
Name: claim_length, dtype: float64

Test Claim Length Summary:
count    1335.000000
mean       13.089888
std         6.840615
min         4.000000
25%         9.000000
50%        10.000000
75%        16.000000
max        63.000000
Name: claim_length, dtype: float64
Visualizations and summaries saved successfully!

Process finished with exit code 0
"""