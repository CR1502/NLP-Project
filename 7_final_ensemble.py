############################################################################
"""
This file tests the FINAL ENSEMBLE there is no need to re-run these files
1_data_cleaning.ipynb
2_Dataset_recreation.py
3_Bagging.ipynb
4_Bagging_Test.py
5_BERT.ipynb
6_BERT_Test.py

# Technically this is the only file that has to be run
"""
############################################################################

import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, BertTokenizerFast
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

device = torch.device("cpu")
print(f"Using device: {device}")

dev_df = pd.read_csv("Final_data/dev_data.csv")
test_df = pd.read_csv("Final_data/test_data.csv")
print("Dev DataFrame shape:", dev_df.shape)
print("Test DataFrame shape:", test_df.shape)

# Extract claim texts and labels
dev_texts = dev_df["claim"].tolist()
y_dev = dev_df["label"].values
test_texts = test_df["claim"].tolist()
y_test = test_df["label"].values

def compute_length(text):
    return len(text.split())

dev_lengths = np.array([compute_length(txt) for txt in dev_texts])
test_lengths = np.array([compute_length(txt) for txt in test_texts])
max_length_dev = dev_lengths.max()
dev_lengths_norm = (dev_lengths / max_length_dev).reshape(-1, 1)
test_lengths_norm = (test_lengths / max_length_dev).reshape(-1, 1)

# Load TF-IDF Vectorizer and Bagging Model Predictions
with open("Compressed model folder/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
X_dev_tfidf = vectorizer.transform(dev_texts)
X_test_tfidf = vectorizer.transform(test_texts)
print("TF-IDF features shape (dev):", X_dev_tfidf.shape)
print("TF-IDF features shape (test):", X_test_tfidf.shape)

with open("Compressed model folder/final_bagging_models.pkl", "rb") as f:
    bagging_models = pickle.load(f)
print("Loaded Bagging models:", list(bagging_models.keys()))

# Compute averaged probability predictions from bagging ensemble
def get_bagging_probs(models, X):
    probs_list = []
    for name, model_bag in models.items():
        p = model_bag.predict_proba(X)
        probs_list.append(p)
    return np.mean(np.stack(probs_list, axis=0), axis=0)

bagging_probs_dev = get_bagging_probs(bagging_models, X_dev_tfidf)
bagging_probs_test = get_bagging_probs(bagging_models, X_test_tfidf)

# Load Final BERT Model and Compute Predictions
tokenizer_final = BertTokenizerFast.from_pretrained("Bert_Model_Final_Tokenizer")
model_final = BertForSequenceClassification.from_pretrained("Bert_Model_Final")
model_final.to(device)
print(f"Final BERT model loaded on {device}!")

def tokenize_texts(texts, tokenizer, max_length=128):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

def get_bert_probs(texts, tokenizer, model, device, max_length=128):
    encodings = tokenize_texts(texts, tokenizer, max_length)
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return probs

bert_probs_dev = get_bert_probs(dev_texts, tokenizer_final, model_final, device, max_length=128)
bert_probs_test = get_bert_probs(test_texts, tokenizer_final, model_final, device, max_length=128)

# Build Meta-Features for Stacking
X_meta_dev = np.hstack([bert_probs_dev, bagging_probs_dev, dev_lengths_norm])
X_meta_test = np.hstack([bert_probs_test, bagging_probs_test, test_lengths_norm])
print("Meta-feature shape (dev):", X_meta_dev.shape)
print("Meta-feature shape (test):", X_meta_test.shape)

# Section 5: Define Meta-Dataset and a Deeper Meta-Model (MLP)
class MetaDataset(Dataset):
    def __init__(self, meta_features, labels):
        self.X = torch.tensor(meta_features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {"features": self.X[idx], "label": self.y[idx]}

meta_dev_dataset = MetaDataset(X_meta_dev, y_dev)
meta_test_dataset = MetaDataset(X_meta_test, y_test)

class ComplexMetaEnsemble(nn.Module):
    def __init__(self, input_dim=5, hidden_dims=[256, 128, 64, 32, 16], dropout_rate=0.3):
        super(ComplexMetaEnsemble, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout_rate)

        self.fc5 = nn.Linear(hidden_dims[3], hidden_dims[4])
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(dropout_rate)

        self.fc6 = nn.Linear(hidden_dims[4], 2)  # Output layer for two weights

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.dropout3(self.relu3(self.fc3(x)))
        x = self.dropout4(self.relu4(self.fc4(x)))
        x = self.dropout5(self.relu5(self.fc5(x)))
        x = self.fc6(x)
        weights = torch.softmax(x, dim=1)
        return weights

meta_model = ComplexMetaEnsemble(input_dim=5, hidden_dims=[256, 128, 128, 64, 32], dropout_rate=0.2).to(device)
print("Deep meta-model initialized.")

from torch.optim import Adam

optimizer = Adam(meta_model.parameters(), lr=3e-05)
loss_fn = nn.NLLLoss()
meta_train_loader = DataLoader(meta_dev_dataset, batch_size=16, shuffle=True)

num_epochs = 500
train_losses = []
train_accuracies = []

meta_model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0
    for batch in meta_train_loader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        weights = meta_model(features)
        p_bert_batch = features[:, :2]
        p_bagging_batch = features[:, 2:4]
        combined_probs = weights[:, 0].unsqueeze(1) * p_bert_batch + weights[:, 1].unsqueeze(1) * p_bagging_batch
        log_probs = torch.log(combined_probs + 1e-8)  # Numerical stability
        loss = loss_fn(log_probs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * features.size(0)
        preds = torch.argmax(combined_probs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss /= total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

print("Deep meta-model training complete.")

# Plot and save training loss curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Train Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Images/training_loss_curve.png", dpi=300)
plt.show()

# Plot and save training accuracy curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), train_accuracies, marker='o', color='green', label='Train Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Images/training_accuracy_curve.png", dpi=300)
plt.show()

# Evaluate the Deep Meta-Model on Test Data
meta_model.eval()
meta_test_loader = DataLoader(meta_test_dataset, batch_size=32, shuffle=False)
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in meta_test_loader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        weights = meta_model(features)
        p_bert_batch = features[:, :2]
        p_bagging_batch = features[:, 2:4]
        combined_probs = weights[:, 0].unsqueeze(1) * p_bert_batch + weights[:, 1].unsqueeze(1) * p_bagging_batch
        preds = torch.argmax(combined_probs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

meta_acc = accuracy_score(all_labels, all_preds)
print("Stacked (Dynamic Weighting) Ensemble Test Accuracy:", meta_acc)
print("Stacked Ensemble Classification Report:\n", classification_report(all_labels, all_preds, zero_division=1))

# Plot and save the test accuracy as a horizontal bar chart
plt.figure(figsize=(6, 6))
plt.barh(["Stacked Ensemble (Dynamic Weighting)"], [meta_acc], color=["purple"])
plt.xlabel("Test Accuracy")
plt.title("Stacked Ensemble Test Accuracy")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig("Images/stacked_ensemble_accuracy.png", dpi=300)
plt.show()

# Compute and save confusion matrix visualization
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix - Stacked Ensemble")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("Images/confusion_matrix_stacked_ensemble.png", dpi=300)
plt.show()

torch.save(meta_model.state_dict(), "Compressed model folder/meta_model_final.pt")
print("Meta-model state saved as 'meta_model_final.pt'.")


############################################################################
# This is the result that I got after running this file
"""
Using device: cpu
Dev DataFrame shape: (2224, 6)
Test DataFrame shape: (1335, 6)
TF-IDF features shape (dev): (2224, 5000)
TF-IDF features shape (test): (1335, 5000)
Loaded Bagging models: ['logreg', 'nb', 'dt', 'svc']
Final BERT model loaded on cpu!
Meta-feature shape (dev): (2224, 5)
Meta-feature shape (test): (1335, 5)
Deep meta-model initialized.
Epoch 1/500 - Loss: 0.4162, Accuracy: 0.8053
Epoch 2/500 - Loss: 0.4147, Accuracy: 0.8049
Epoch 3/500 - Loss: 0.4123, Accuracy: 0.8067
Epoch 4/500 - Loss: 0.4071, Accuracy: 0.8062
Epoch 5/500 - Loss: 0.3992, Accuracy: 0.8031
Epoch 6/500 - Loss: 0.3949, Accuracy: 0.8058
Epoch 7/500 - Loss: 0.3936, Accuracy: 0.8031
Epoch 8/500 - Loss: 0.3945, Accuracy: 0.8026
Epoch 9/500 - Loss: 0.3937, Accuracy: 0.8040
Epoch 10/500 - Loss: 0.3944, Accuracy: 0.8031
Epoch 11/500 - Loss: 0.3947, Accuracy: 0.8049
Epoch 12/500 - Loss: 0.3929, Accuracy: 0.8031
Epoch 13/500 - Loss: 0.3934, Accuracy: 0.8049
Epoch 14/500 - Loss: 0.3932, Accuracy: 0.8035
Epoch 15/500 - Loss: 0.3924, Accuracy: 0.8044
Epoch 16/500 - Loss: 0.3924, Accuracy: 0.8053
Epoch 17/500 - Loss: 0.3938, Accuracy: 0.8044
Epoch 18/500 - Loss: 0.3906, Accuracy: 0.8040
Epoch 19/500 - Loss: 0.3938, Accuracy: 0.8040
Epoch 20/500 - Loss: 0.3933, Accuracy: 0.8058
Epoch 21/500 - Loss: 0.3929, Accuracy: 0.8049
Epoch 22/500 - Loss: 0.3921, Accuracy: 0.8026
Epoch 23/500 - Loss: 0.3921, Accuracy: 0.8031
Epoch 24/500 - Loss: 0.3912, Accuracy: 0.8040
Epoch 25/500 - Loss: 0.3925, Accuracy: 0.8044
Epoch 26/500 - Loss: 0.3931, Accuracy: 0.8035
Epoch 27/500 - Loss: 0.3925, Accuracy: 0.8026
Epoch 28/500 - Loss: 0.3928, Accuracy: 0.8022
Epoch 29/500 - Loss: 0.3908, Accuracy: 0.8031
Epoch 30/500 - Loss: 0.3912, Accuracy: 0.8049
Epoch 31/500 - Loss: 0.3907, Accuracy: 0.8022
Epoch 32/500 - Loss: 0.3906, Accuracy: 0.8026
Epoch 33/500 - Loss: 0.3893, Accuracy: 0.8049
Epoch 34/500 - Loss: 0.3904, Accuracy: 0.8026
Epoch 35/500 - Loss: 0.3891, Accuracy: 0.8044
Epoch 36/500 - Loss: 0.3903, Accuracy: 0.8058
Epoch 37/500 - Loss: 0.3903, Accuracy: 0.8044
Epoch 38/500 - Loss: 0.3902, Accuracy: 0.8040
Epoch 39/500 - Loss: 0.3897, Accuracy: 0.8053
Epoch 40/500 - Loss: 0.3900, Accuracy: 0.8022
Epoch 41/500 - Loss: 0.3904, Accuracy: 0.8035
Epoch 42/500 - Loss: 0.3891, Accuracy: 0.8040
Epoch 43/500 - Loss: 0.3877, Accuracy: 0.8035
Epoch 44/500 - Loss: 0.3887, Accuracy: 0.8013
Epoch 45/500 - Loss: 0.3880, Accuracy: 0.8013
Epoch 46/500 - Loss: 0.3884, Accuracy: 0.8035
Epoch 47/500 - Loss: 0.3878, Accuracy: 0.8022
Epoch 48/500 - Loss: 0.3889, Accuracy: 0.8017
Epoch 49/500 - Loss: 0.3892, Accuracy: 0.8053
Epoch 50/500 - Loss: 0.3877, Accuracy: 0.8031
Epoch 51/500 - Loss: 0.3888, Accuracy: 0.8031
Epoch 52/500 - Loss: 0.3878, Accuracy: 0.8031
Epoch 53/500 - Loss: 0.3868, Accuracy: 0.8026
Epoch 54/500 - Loss: 0.3884, Accuracy: 0.8017
Epoch 55/500 - Loss: 0.3875, Accuracy: 0.8017
Epoch 56/500 - Loss: 0.3868, Accuracy: 0.8044
Epoch 57/500 - Loss: 0.3856, Accuracy: 0.8013
Epoch 58/500 - Loss: 0.3883, Accuracy: 0.8026
Epoch 59/500 - Loss: 0.3863, Accuracy: 0.8013
Epoch 60/500 - Loss: 0.3871, Accuracy: 0.8040
Epoch 61/500 - Loss: 0.3871, Accuracy: 0.8013
Epoch 62/500 - Loss: 0.3883, Accuracy: 0.8035
Epoch 63/500 - Loss: 0.3855, Accuracy: 0.8026
Epoch 64/500 - Loss: 0.3863, Accuracy: 0.8044
Epoch 65/500 - Loss: 0.3858, Accuracy: 0.8031
Epoch 66/500 - Loss: 0.3859, Accuracy: 0.8026
Epoch 67/500 - Loss: 0.3865, Accuracy: 0.8022
Epoch 68/500 - Loss: 0.3861, Accuracy: 0.8026
Epoch 69/500 - Loss: 0.3863, Accuracy: 0.8013
Epoch 70/500 - Loss: 0.3867, Accuracy: 0.7995
Epoch 71/500 - Loss: 0.3853, Accuracy: 0.8026
Epoch 72/500 - Loss: 0.3851, Accuracy: 0.8004
Epoch 73/500 - Loss: 0.3855, Accuracy: 0.8017
Epoch 74/500 - Loss: 0.3862, Accuracy: 0.8031
Epoch 75/500 - Loss: 0.3848, Accuracy: 0.8031
Epoch 76/500 - Loss: 0.3858, Accuracy: 0.8026
Epoch 77/500 - Loss: 0.3868, Accuracy: 0.8026
Epoch 78/500 - Loss: 0.3863, Accuracy: 0.8008
Epoch 79/500 - Loss: 0.3857, Accuracy: 0.8017
Epoch 80/500 - Loss: 0.3852, Accuracy: 0.8004
Epoch 81/500 - Loss: 0.3837, Accuracy: 0.8008
Epoch 82/500 - Loss: 0.3848, Accuracy: 0.8035
Epoch 83/500 - Loss: 0.3842, Accuracy: 0.8031
Epoch 84/500 - Loss: 0.3861, Accuracy: 0.8008
Epoch 85/500 - Loss: 0.3859, Accuracy: 0.8040
Epoch 86/500 - Loss: 0.3843, Accuracy: 0.8031
Epoch 87/500 - Loss: 0.3858, Accuracy: 0.8004
Epoch 88/500 - Loss: 0.3858, Accuracy: 0.8022
Epoch 89/500 - Loss: 0.3842, Accuracy: 0.8022
Epoch 90/500 - Loss: 0.3837, Accuracy: 0.8022
Epoch 91/500 - Loss: 0.3830, Accuracy: 0.8022
Epoch 92/500 - Loss: 0.3856, Accuracy: 0.8013
Epoch 93/500 - Loss: 0.3844, Accuracy: 0.8026
Epoch 94/500 - Loss: 0.3856, Accuracy: 0.8008
Epoch 95/500 - Loss: 0.3839, Accuracy: 0.8031
Epoch 96/500 - Loss: 0.3848, Accuracy: 0.8040
Epoch 97/500 - Loss: 0.3844, Accuracy: 0.8035
Epoch 98/500 - Loss: 0.3863, Accuracy: 0.8013
Epoch 99/500 - Loss: 0.3847, Accuracy: 0.8035
Epoch 100/500 - Loss: 0.3840, Accuracy: 0.7990
Epoch 101/500 - Loss: 0.3830, Accuracy: 0.8017
Epoch 102/500 - Loss: 0.3860, Accuracy: 0.8040
Epoch 103/500 - Loss: 0.3851, Accuracy: 0.8026
Epoch 104/500 - Loss: 0.3863, Accuracy: 0.8022
Epoch 105/500 - Loss: 0.3854, Accuracy: 0.8044
Epoch 106/500 - Loss: 0.3844, Accuracy: 0.8017
Epoch 107/500 - Loss: 0.3840, Accuracy: 0.8049
Epoch 108/500 - Loss: 0.3854, Accuracy: 0.8031
Epoch 109/500 - Loss: 0.3839, Accuracy: 0.8058
Epoch 110/500 - Loss: 0.3836, Accuracy: 0.8035
Epoch 111/500 - Loss: 0.3853, Accuracy: 0.8044
Epoch 112/500 - Loss: 0.3845, Accuracy: 0.8040
Epoch 113/500 - Loss: 0.3856, Accuracy: 0.7999
Epoch 114/500 - Loss: 0.3837, Accuracy: 0.8022
Epoch 115/500 - Loss: 0.3834, Accuracy: 0.8017
Epoch 116/500 - Loss: 0.3825, Accuracy: 0.8035
Epoch 117/500 - Loss: 0.3824, Accuracy: 0.8013
Epoch 118/500 - Loss: 0.3845, Accuracy: 0.8008
Epoch 119/500 - Loss: 0.3855, Accuracy: 0.8040
Epoch 120/500 - Loss: 0.3836, Accuracy: 0.8017
Epoch 121/500 - Loss: 0.3843, Accuracy: 0.8031
Epoch 122/500 - Loss: 0.3840, Accuracy: 0.8040
Epoch 123/500 - Loss: 0.3841, Accuracy: 0.8040
Epoch 124/500 - Loss: 0.3845, Accuracy: 0.8035
Epoch 125/500 - Loss: 0.3839, Accuracy: 0.8035
Epoch 126/500 - Loss: 0.3846, Accuracy: 0.8017
Epoch 127/500 - Loss: 0.3831, Accuracy: 0.8022
Epoch 128/500 - Loss: 0.3845, Accuracy: 0.8049
Epoch 129/500 - Loss: 0.3847, Accuracy: 0.8040
Epoch 130/500 - Loss: 0.3842, Accuracy: 0.8004
Epoch 131/500 - Loss: 0.3856, Accuracy: 0.8017
Epoch 132/500 - Loss: 0.3832, Accuracy: 0.8040
Epoch 133/500 - Loss: 0.3838, Accuracy: 0.8044
Epoch 134/500 - Loss: 0.3843, Accuracy: 0.8031
Epoch 135/500 - Loss: 0.3829, Accuracy: 0.8049
Epoch 136/500 - Loss: 0.3833, Accuracy: 0.8026
Epoch 137/500 - Loss: 0.3835, Accuracy: 0.8026
Epoch 138/500 - Loss: 0.3835, Accuracy: 0.8031
Epoch 139/500 - Loss: 0.3827, Accuracy: 0.8044
Epoch 140/500 - Loss: 0.3844, Accuracy: 0.8040
Epoch 141/500 - Loss: 0.3845, Accuracy: 0.8040
Epoch 142/500 - Loss: 0.3842, Accuracy: 0.8053
Epoch 143/500 - Loss: 0.3836, Accuracy: 0.8040
Epoch 144/500 - Loss: 0.3833, Accuracy: 0.8062
Epoch 145/500 - Loss: 0.3832, Accuracy: 0.8022
Epoch 146/500 - Loss: 0.3836, Accuracy: 0.8035
Epoch 147/500 - Loss: 0.3832, Accuracy: 0.8049
Epoch 148/500 - Loss: 0.3842, Accuracy: 0.8022
Epoch 149/500 - Loss: 0.3830, Accuracy: 0.8022
Epoch 150/500 - Loss: 0.3840, Accuracy: 0.8035
Epoch 151/500 - Loss: 0.3830, Accuracy: 0.8062
Epoch 152/500 - Loss: 0.3837, Accuracy: 0.8035
Epoch 153/500 - Loss: 0.3830, Accuracy: 0.8040
Epoch 154/500 - Loss: 0.3835, Accuracy: 0.8035
Epoch 155/500 - Loss: 0.3823, Accuracy: 0.8053
Epoch 156/500 - Loss: 0.3830, Accuracy: 0.8031
Epoch 157/500 - Loss: 0.3827, Accuracy: 0.8026
Epoch 158/500 - Loss: 0.3811, Accuracy: 0.8062
Epoch 159/500 - Loss: 0.3831, Accuracy: 0.8035
Epoch 160/500 - Loss: 0.3847, Accuracy: 0.8053
Epoch 161/500 - Loss: 0.3835, Accuracy: 0.8035
Epoch 162/500 - Loss: 0.3835, Accuracy: 0.8049
Epoch 163/500 - Loss: 0.3834, Accuracy: 0.8058
Epoch 164/500 - Loss: 0.3837, Accuracy: 0.8058
Epoch 165/500 - Loss: 0.3822, Accuracy: 0.8031
Epoch 166/500 - Loss: 0.3828, Accuracy: 0.8008
Epoch 167/500 - Loss: 0.3818, Accuracy: 0.8058
Epoch 168/500 - Loss: 0.3819, Accuracy: 0.8026
Epoch 169/500 - Loss: 0.3830, Accuracy: 0.8040
Epoch 170/500 - Loss: 0.3826, Accuracy: 0.8044
Epoch 171/500 - Loss: 0.3837, Accuracy: 0.8026
Epoch 172/500 - Loss: 0.3829, Accuracy: 0.8044
Epoch 173/500 - Loss: 0.3817, Accuracy: 0.8017
Epoch 174/500 - Loss: 0.3830, Accuracy: 0.8067
Epoch 175/500 - Loss: 0.3840, Accuracy: 0.8062
Epoch 176/500 - Loss: 0.3842, Accuracy: 0.8022
Epoch 177/500 - Loss: 0.3841, Accuracy: 0.8049
Epoch 178/500 - Loss: 0.3814, Accuracy: 0.8040
Epoch 179/500 - Loss: 0.3843, Accuracy: 0.8031
Epoch 180/500 - Loss: 0.3842, Accuracy: 0.8035
Epoch 181/500 - Loss: 0.3836, Accuracy: 0.8049
Epoch 182/500 - Loss: 0.3815, Accuracy: 0.8040
Epoch 183/500 - Loss: 0.3824, Accuracy: 0.8040
Epoch 184/500 - Loss: 0.3844, Accuracy: 0.8053
Epoch 185/500 - Loss: 0.3821, Accuracy: 0.8058
Epoch 186/500 - Loss: 0.3811, Accuracy: 0.8085
Epoch 187/500 - Loss: 0.3820, Accuracy: 0.8049
Epoch 188/500 - Loss: 0.3826, Accuracy: 0.8044
Epoch 189/500 - Loss: 0.3823, Accuracy: 0.8067
Epoch 190/500 - Loss: 0.3832, Accuracy: 0.8049
Epoch 191/500 - Loss: 0.3840, Accuracy: 0.8053
Epoch 192/500 - Loss: 0.3818, Accuracy: 0.8062
Epoch 193/500 - Loss: 0.3833, Accuracy: 0.8071
Epoch 194/500 - Loss: 0.3822, Accuracy: 0.8035
Epoch 195/500 - Loss: 0.3828, Accuracy: 0.8062
Epoch 196/500 - Loss: 0.3825, Accuracy: 0.8076
Epoch 197/500 - Loss: 0.3824, Accuracy: 0.8080
Epoch 198/500 - Loss: 0.3826, Accuracy: 0.8058
Epoch 199/500 - Loss: 0.3841, Accuracy: 0.8026
Epoch 200/500 - Loss: 0.3829, Accuracy: 0.8058
Epoch 201/500 - Loss: 0.3828, Accuracy: 0.8058
Epoch 202/500 - Loss: 0.3813, Accuracy: 0.8053
Epoch 203/500 - Loss: 0.3820, Accuracy: 0.8058
Epoch 204/500 - Loss: 0.3820, Accuracy: 0.8067
Epoch 205/500 - Loss: 0.3824, Accuracy: 0.8071
Epoch 206/500 - Loss: 0.3817, Accuracy: 0.8053
Epoch 207/500 - Loss: 0.3813, Accuracy: 0.8080
Epoch 208/500 - Loss: 0.3814, Accuracy: 0.8053
Epoch 209/500 - Loss: 0.3820, Accuracy: 0.8049
Epoch 210/500 - Loss: 0.3818, Accuracy: 0.8049
Epoch 211/500 - Loss: 0.3823, Accuracy: 0.8058
Epoch 212/500 - Loss: 0.3814, Accuracy: 0.8040
Epoch 213/500 - Loss: 0.3826, Accuracy: 0.8076
Epoch 214/500 - Loss: 0.3828, Accuracy: 0.8071
Epoch 215/500 - Loss: 0.3821, Accuracy: 0.8062
Epoch 216/500 - Loss: 0.3829, Accuracy: 0.8062
Epoch 217/500 - Loss: 0.3834, Accuracy: 0.8085
Epoch 218/500 - Loss: 0.3842, Accuracy: 0.8053
Epoch 219/500 - Loss: 0.3820, Accuracy: 0.8080
Epoch 220/500 - Loss: 0.3833, Accuracy: 0.8049
Epoch 221/500 - Loss: 0.3817, Accuracy: 0.8058
Epoch 222/500 - Loss: 0.3825, Accuracy: 0.8067
Epoch 223/500 - Loss: 0.3832, Accuracy: 0.8058
Epoch 224/500 - Loss: 0.3813, Accuracy: 0.8071
Epoch 225/500 - Loss: 0.3816, Accuracy: 0.8044
Epoch 226/500 - Loss: 0.3813, Accuracy: 0.8062
Epoch 227/500 - Loss: 0.3810, Accuracy: 0.8058
Epoch 228/500 - Loss: 0.3830, Accuracy: 0.8044
Epoch 229/500 - Loss: 0.3795, Accuracy: 0.8062
Epoch 230/500 - Loss: 0.3831, Accuracy: 0.8053
Epoch 231/500 - Loss: 0.3811, Accuracy: 0.8067
Epoch 232/500 - Loss: 0.3816, Accuracy: 0.8085
Epoch 233/500 - Loss: 0.3819, Accuracy: 0.8031
Epoch 234/500 - Loss: 0.3813, Accuracy: 0.8058
Epoch 235/500 - Loss: 0.3839, Accuracy: 0.8067
Epoch 236/500 - Loss: 0.3806, Accuracy: 0.8089
Epoch 237/500 - Loss: 0.3808, Accuracy: 0.8085
Epoch 238/500 - Loss: 0.3810, Accuracy: 0.8080
Epoch 239/500 - Loss: 0.3817, Accuracy: 0.8094
Epoch 240/500 - Loss: 0.3828, Accuracy: 0.8031
Epoch 241/500 - Loss: 0.3815, Accuracy: 0.8076
Epoch 242/500 - Loss: 0.3810, Accuracy: 0.8031
Epoch 243/500 - Loss: 0.3816, Accuracy: 0.8062
Epoch 244/500 - Loss: 0.3818, Accuracy: 0.8076
Epoch 245/500 - Loss: 0.3795, Accuracy: 0.8067
Epoch 246/500 - Loss: 0.3812, Accuracy: 0.8071
Epoch 247/500 - Loss: 0.3830, Accuracy: 0.8062
Epoch 248/500 - Loss: 0.3818, Accuracy: 0.8098
Epoch 249/500 - Loss: 0.3831, Accuracy: 0.8085
Epoch 250/500 - Loss: 0.3813, Accuracy: 0.8044
Epoch 251/500 - Loss: 0.3824, Accuracy: 0.8044
Epoch 252/500 - Loss: 0.3814, Accuracy: 0.8080
Epoch 253/500 - Loss: 0.3815, Accuracy: 0.8062
Epoch 254/500 - Loss: 0.3798, Accuracy: 0.8085
Epoch 255/500 - Loss: 0.3806, Accuracy: 0.8035
Epoch 256/500 - Loss: 0.3807, Accuracy: 0.8040
Epoch 257/500 - Loss: 0.3804, Accuracy: 0.8112
Epoch 258/500 - Loss: 0.3830, Accuracy: 0.8062
Epoch 259/500 - Loss: 0.3809, Accuracy: 0.8076
Epoch 260/500 - Loss: 0.3830, Accuracy: 0.8094
Epoch 261/500 - Loss: 0.3823, Accuracy: 0.8049
Epoch 262/500 - Loss: 0.3818, Accuracy: 0.8094
Epoch 263/500 - Loss: 0.3814, Accuracy: 0.8071
Epoch 264/500 - Loss: 0.3805, Accuracy: 0.8071
Epoch 265/500 - Loss: 0.3814, Accuracy: 0.8067
Epoch 266/500 - Loss: 0.3822, Accuracy: 0.8062
Epoch 267/500 - Loss: 0.3817, Accuracy: 0.8044
Epoch 268/500 - Loss: 0.3812, Accuracy: 0.8080
Epoch 269/500 - Loss: 0.3822, Accuracy: 0.8076
Epoch 270/500 - Loss: 0.3816, Accuracy: 0.8071
Epoch 271/500 - Loss: 0.3808, Accuracy: 0.8089
Epoch 272/500 - Loss: 0.3799, Accuracy: 0.8085
Epoch 273/500 - Loss: 0.3809, Accuracy: 0.8044
Epoch 274/500 - Loss: 0.3823, Accuracy: 0.8053
Epoch 275/500 - Loss: 0.3816, Accuracy: 0.8053
Epoch 276/500 - Loss: 0.3810, Accuracy: 0.8107
Epoch 277/500 - Loss: 0.3807, Accuracy: 0.8094
Epoch 278/500 - Loss: 0.3831, Accuracy: 0.8112
Epoch 279/500 - Loss: 0.3806, Accuracy: 0.8058
Epoch 280/500 - Loss: 0.3806, Accuracy: 0.8094
Epoch 281/500 - Loss: 0.3812, Accuracy: 0.8067
Epoch 282/500 - Loss: 0.3816, Accuracy: 0.8067
Epoch 283/500 - Loss: 0.3790, Accuracy: 0.8071
Epoch 284/500 - Loss: 0.3806, Accuracy: 0.8080
Epoch 285/500 - Loss: 0.3807, Accuracy: 0.8098
Epoch 286/500 - Loss: 0.3803, Accuracy: 0.8058
Epoch 287/500 - Loss: 0.3815, Accuracy: 0.8080
Epoch 288/500 - Loss: 0.3802, Accuracy: 0.8085
Epoch 289/500 - Loss: 0.3796, Accuracy: 0.8067
Epoch 290/500 - Loss: 0.3815, Accuracy: 0.8067
Epoch 291/500 - Loss: 0.3813, Accuracy: 0.8076
Epoch 292/500 - Loss: 0.3810, Accuracy: 0.8071
Epoch 293/500 - Loss: 0.3802, Accuracy: 0.8094
Epoch 294/500 - Loss: 0.3811, Accuracy: 0.8080
Epoch 295/500 - Loss: 0.3800, Accuracy: 0.8089
Epoch 296/500 - Loss: 0.3797, Accuracy: 0.8076
Epoch 297/500 - Loss: 0.3812, Accuracy: 0.8085
Epoch 298/500 - Loss: 0.3811, Accuracy: 0.8067
Epoch 299/500 - Loss: 0.3812, Accuracy: 0.8053
Epoch 300/500 - Loss: 0.3822, Accuracy: 0.8058
Epoch 301/500 - Loss: 0.3788, Accuracy: 0.8094
Epoch 302/500 - Loss: 0.3800, Accuracy: 0.8098
Epoch 303/500 - Loss: 0.3812, Accuracy: 0.8062
Epoch 304/500 - Loss: 0.3817, Accuracy: 0.8089
Epoch 305/500 - Loss: 0.3796, Accuracy: 0.8071
Epoch 306/500 - Loss: 0.3813, Accuracy: 0.8080
Epoch 307/500 - Loss: 0.3811, Accuracy: 0.8085
Epoch 308/500 - Loss: 0.3809, Accuracy: 0.8076
Epoch 309/500 - Loss: 0.3807, Accuracy: 0.8062
Epoch 310/500 - Loss: 0.3811, Accuracy: 0.8089
Epoch 311/500 - Loss: 0.3794, Accuracy: 0.8085
Epoch 312/500 - Loss: 0.3803, Accuracy: 0.8067
Epoch 313/500 - Loss: 0.3789, Accuracy: 0.8098
Epoch 314/500 - Loss: 0.3813, Accuracy: 0.8089
Epoch 315/500 - Loss: 0.3803, Accuracy: 0.8080
Epoch 316/500 - Loss: 0.3811, Accuracy: 0.8089
Epoch 317/500 - Loss: 0.3798, Accuracy: 0.8080
Epoch 318/500 - Loss: 0.3786, Accuracy: 0.8062
Epoch 319/500 - Loss: 0.3813, Accuracy: 0.8107
Epoch 320/500 - Loss: 0.3806, Accuracy: 0.8058
Epoch 321/500 - Loss: 0.3796, Accuracy: 0.8094
Epoch 322/500 - Loss: 0.3810, Accuracy: 0.8049
Epoch 323/500 - Loss: 0.3792, Accuracy: 0.8094
Epoch 324/500 - Loss: 0.3797, Accuracy: 0.8071
Epoch 325/500 - Loss: 0.3808, Accuracy: 0.8098
Epoch 326/500 - Loss: 0.3828, Accuracy: 0.8076
Epoch 327/500 - Loss: 0.3804, Accuracy: 0.8094
Epoch 328/500 - Loss: 0.3811, Accuracy: 0.8098
Epoch 329/500 - Loss: 0.3809, Accuracy: 0.8076
Epoch 330/500 - Loss: 0.3819, Accuracy: 0.8089
Epoch 331/500 - Loss: 0.3787, Accuracy: 0.8085
Epoch 332/500 - Loss: 0.3820, Accuracy: 0.8089
Epoch 333/500 - Loss: 0.3804, Accuracy: 0.8076
Epoch 334/500 - Loss: 0.3816, Accuracy: 0.8080
Epoch 335/500 - Loss: 0.3777, Accuracy: 0.8080
Epoch 336/500 - Loss: 0.3801, Accuracy: 0.8094
Epoch 337/500 - Loss: 0.3796, Accuracy: 0.8080
Epoch 338/500 - Loss: 0.3785, Accuracy: 0.8098
Epoch 339/500 - Loss: 0.3787, Accuracy: 0.8089
Epoch 340/500 - Loss: 0.3803, Accuracy: 0.8103
Epoch 341/500 - Loss: 0.3794, Accuracy: 0.8076
Epoch 342/500 - Loss: 0.3811, Accuracy: 0.8094
Epoch 343/500 - Loss: 0.3802, Accuracy: 0.8080
Epoch 344/500 - Loss: 0.3805, Accuracy: 0.8080
Epoch 345/500 - Loss: 0.3803, Accuracy: 0.8085
Epoch 346/500 - Loss: 0.3803, Accuracy: 0.8098
Epoch 347/500 - Loss: 0.3800, Accuracy: 0.8071
Epoch 348/500 - Loss: 0.3781, Accuracy: 0.8107
Epoch 349/500 - Loss: 0.3799, Accuracy: 0.8107
Epoch 350/500 - Loss: 0.3810, Accuracy: 0.8076
Epoch 351/500 - Loss: 0.3794, Accuracy: 0.8089
Epoch 352/500 - Loss: 0.3800, Accuracy: 0.8080
Epoch 353/500 - Loss: 0.3791, Accuracy: 0.8125
Epoch 354/500 - Loss: 0.3789, Accuracy: 0.8067
Epoch 355/500 - Loss: 0.3803, Accuracy: 0.8067
Epoch 356/500 - Loss: 0.3794, Accuracy: 0.8112
Epoch 357/500 - Loss: 0.3817, Accuracy: 0.8062
Epoch 358/500 - Loss: 0.3787, Accuracy: 0.8098
Epoch 359/500 - Loss: 0.3802, Accuracy: 0.8076
Epoch 360/500 - Loss: 0.3794, Accuracy: 0.8112
Epoch 361/500 - Loss: 0.3809, Accuracy: 0.8103
Epoch 362/500 - Loss: 0.3793, Accuracy: 0.8067
Epoch 363/500 - Loss: 0.3804, Accuracy: 0.8071
Epoch 364/500 - Loss: 0.3798, Accuracy: 0.8094
Epoch 365/500 - Loss: 0.3777, Accuracy: 0.8103
Epoch 366/500 - Loss: 0.3790, Accuracy: 0.8080
Epoch 367/500 - Loss: 0.3782, Accuracy: 0.8098
Epoch 368/500 - Loss: 0.3794, Accuracy: 0.8089
Epoch 369/500 - Loss: 0.3789, Accuracy: 0.8071
Epoch 370/500 - Loss: 0.3793, Accuracy: 0.8085
Epoch 371/500 - Loss: 0.3787, Accuracy: 0.8103
Epoch 372/500 - Loss: 0.3792, Accuracy: 0.8085
Epoch 373/500 - Loss: 0.3799, Accuracy: 0.8058
Epoch 374/500 - Loss: 0.3788, Accuracy: 0.8062
Epoch 375/500 - Loss: 0.3794, Accuracy: 0.8103
Epoch 376/500 - Loss: 0.3788, Accuracy: 0.8071
Epoch 377/500 - Loss: 0.3789, Accuracy: 0.8089
Epoch 378/500 - Loss: 0.3791, Accuracy: 0.8071
Epoch 379/500 - Loss: 0.3797, Accuracy: 0.8085
Epoch 380/500 - Loss: 0.3819, Accuracy: 0.8076
Epoch 381/500 - Loss: 0.3801, Accuracy: 0.8080
Epoch 382/500 - Loss: 0.3798, Accuracy: 0.8089
Epoch 383/500 - Loss: 0.3793, Accuracy: 0.8076
Epoch 384/500 - Loss: 0.3790, Accuracy: 0.8076
Epoch 385/500 - Loss: 0.3785, Accuracy: 0.8080
Epoch 386/500 - Loss: 0.3795, Accuracy: 0.8094
Epoch 387/500 - Loss: 0.3787, Accuracy: 0.8098
Epoch 388/500 - Loss: 0.3797, Accuracy: 0.8080
Epoch 389/500 - Loss: 0.3773, Accuracy: 0.8103
Epoch 390/500 - Loss: 0.3782, Accuracy: 0.8085
Epoch 391/500 - Loss: 0.3775, Accuracy: 0.8098
Epoch 392/500 - Loss: 0.3795, Accuracy: 0.8080
Epoch 393/500 - Loss: 0.3791, Accuracy: 0.8107
Epoch 394/500 - Loss: 0.3796, Accuracy: 0.8080
Epoch 395/500 - Loss: 0.3789, Accuracy: 0.8094
Epoch 396/500 - Loss: 0.3798, Accuracy: 0.8080
Epoch 397/500 - Loss: 0.3780, Accuracy: 0.8071
Epoch 398/500 - Loss: 0.3768, Accuracy: 0.8076
Epoch 399/500 - Loss: 0.3790, Accuracy: 0.8076
Epoch 400/500 - Loss: 0.3794, Accuracy: 0.8080
Epoch 401/500 - Loss: 0.3779, Accuracy: 0.8076
Epoch 402/500 - Loss: 0.3799, Accuracy: 0.8089
Epoch 403/500 - Loss: 0.3807, Accuracy: 0.8103
Epoch 404/500 - Loss: 0.3797, Accuracy: 0.8071
Epoch 405/500 - Loss: 0.3774, Accuracy: 0.8080
Epoch 406/500 - Loss: 0.3797, Accuracy: 0.8085
Epoch 407/500 - Loss: 0.3777, Accuracy: 0.8094
Epoch 408/500 - Loss: 0.3800, Accuracy: 0.8098
Epoch 409/500 - Loss: 0.3781, Accuracy: 0.8103
Epoch 410/500 - Loss: 0.3780, Accuracy: 0.8080
Epoch 411/500 - Loss: 0.3789, Accuracy: 0.8058
Epoch 412/500 - Loss: 0.3782, Accuracy: 0.8089
Epoch 413/500 - Loss: 0.3795, Accuracy: 0.8094
Epoch 414/500 - Loss: 0.3790, Accuracy: 0.8112
Epoch 415/500 - Loss: 0.3782, Accuracy: 0.8098
Epoch 416/500 - Loss: 0.3773, Accuracy: 0.8107
Epoch 417/500 - Loss: 0.3772, Accuracy: 0.8071
Epoch 418/500 - Loss: 0.3773, Accuracy: 0.8067
Epoch 419/500 - Loss: 0.3795, Accuracy: 0.8071
Epoch 420/500 - Loss: 0.3763, Accuracy: 0.8098
Epoch 421/500 - Loss: 0.3774, Accuracy: 0.8067
Epoch 422/500 - Loss: 0.3790, Accuracy: 0.8058
Epoch 423/500 - Loss: 0.3786, Accuracy: 0.8067
Epoch 424/500 - Loss: 0.3784, Accuracy: 0.8107
Epoch 425/500 - Loss: 0.3759, Accuracy: 0.8085
Epoch 426/500 - Loss: 0.3772, Accuracy: 0.8076
Epoch 427/500 - Loss: 0.3769, Accuracy: 0.8053
Epoch 428/500 - Loss: 0.3787, Accuracy: 0.8098
Epoch 429/500 - Loss: 0.3790, Accuracy: 0.8089
Epoch 430/500 - Loss: 0.3780, Accuracy: 0.8053
Epoch 431/500 - Loss: 0.3772, Accuracy: 0.8085
Epoch 432/500 - Loss: 0.3758, Accuracy: 0.8094
Epoch 433/500 - Loss: 0.3790, Accuracy: 0.8098
Epoch 434/500 - Loss: 0.3770, Accuracy: 0.8089
Epoch 435/500 - Loss: 0.3779, Accuracy: 0.8094
Epoch 436/500 - Loss: 0.3777, Accuracy: 0.8049
Epoch 437/500 - Loss: 0.3786, Accuracy: 0.8103
Epoch 438/500 - Loss: 0.3777, Accuracy: 0.8116
Epoch 439/500 - Loss: 0.3780, Accuracy: 0.8080
Epoch 440/500 - Loss: 0.3793, Accuracy: 0.8094
Epoch 441/500 - Loss: 0.3774, Accuracy: 0.8062
Epoch 442/500 - Loss: 0.3796, Accuracy: 0.8071
Epoch 443/500 - Loss: 0.3774, Accuracy: 0.8076
Epoch 444/500 - Loss: 0.3779, Accuracy: 0.8089
Epoch 445/500 - Loss: 0.3765, Accuracy: 0.8067
Epoch 446/500 - Loss: 0.3777, Accuracy: 0.8098
Epoch 447/500 - Loss: 0.3795, Accuracy: 0.8085
Epoch 448/500 - Loss: 0.3794, Accuracy: 0.8089
Epoch 449/500 - Loss: 0.3800, Accuracy: 0.8080
Epoch 450/500 - Loss: 0.3783, Accuracy: 0.8094
Epoch 451/500 - Loss: 0.3774, Accuracy: 0.8071
Epoch 452/500 - Loss: 0.3766, Accuracy: 0.8116
Epoch 453/500 - Loss: 0.3759, Accuracy: 0.8098
Epoch 454/500 - Loss: 0.3777, Accuracy: 0.8103
Epoch 455/500 - Loss: 0.3777, Accuracy: 0.8076
Epoch 456/500 - Loss: 0.3776, Accuracy: 0.8085
Epoch 457/500 - Loss: 0.3791, Accuracy: 0.8085
Epoch 458/500 - Loss: 0.3800, Accuracy: 0.8094
Epoch 459/500 - Loss: 0.3784, Accuracy: 0.8094
Epoch 460/500 - Loss: 0.3771, Accuracy: 0.8089
Epoch 461/500 - Loss: 0.3790, Accuracy: 0.8085
Epoch 462/500 - Loss: 0.3781, Accuracy: 0.8094
Epoch 463/500 - Loss: 0.3776, Accuracy: 0.8058
Epoch 464/500 - Loss: 0.3767, Accuracy: 0.8076
Epoch 465/500 - Loss: 0.3761, Accuracy: 0.8094
Epoch 466/500 - Loss: 0.3791, Accuracy: 0.8076
Epoch 467/500 - Loss: 0.3774, Accuracy: 0.8094
Epoch 468/500 - Loss: 0.3776, Accuracy: 0.8076
Epoch 469/500 - Loss: 0.3792, Accuracy: 0.8085
Epoch 470/500 - Loss: 0.3779, Accuracy: 0.8089
Epoch 471/500 - Loss: 0.3759, Accuracy: 0.8098
Epoch 472/500 - Loss: 0.3785, Accuracy: 0.8049
Epoch 473/500 - Loss: 0.3788, Accuracy: 0.8076
Epoch 474/500 - Loss: 0.3769, Accuracy: 0.8085
Epoch 475/500 - Loss: 0.3784, Accuracy: 0.8103
Epoch 476/500 - Loss: 0.3770, Accuracy: 0.8058
Epoch 477/500 - Loss: 0.3780, Accuracy: 0.8094
Epoch 478/500 - Loss: 0.3779, Accuracy: 0.8089
Epoch 479/500 - Loss: 0.3764, Accuracy: 0.8085
Epoch 480/500 - Loss: 0.3787, Accuracy: 0.8049
Epoch 481/500 - Loss: 0.3769, Accuracy: 0.8098
Epoch 482/500 - Loss: 0.3768, Accuracy: 0.8071
Epoch 483/500 - Loss: 0.3771, Accuracy: 0.8089
Epoch 484/500 - Loss: 0.3784, Accuracy: 0.8085
Epoch 485/500 - Loss: 0.3773, Accuracy: 0.8103
Epoch 486/500 - Loss: 0.3780, Accuracy: 0.8085
Epoch 487/500 - Loss: 0.3770, Accuracy: 0.8094
Epoch 488/500 - Loss: 0.3764, Accuracy: 0.8094
Epoch 489/500 - Loss: 0.3797, Accuracy: 0.8076
Epoch 490/500 - Loss: 0.3755, Accuracy: 0.8085
Epoch 491/500 - Loss: 0.3777, Accuracy: 0.8076
Epoch 492/500 - Loss: 0.3762, Accuracy: 0.8103
Epoch 493/500 - Loss: 0.3768, Accuracy: 0.8116
Epoch 494/500 - Loss: 0.3763, Accuracy: 0.8098
Epoch 495/500 - Loss: 0.3784, Accuracy: 0.8098
Epoch 496/500 - Loss: 0.3766, Accuracy: 0.8103
Epoch 497/500 - Loss: 0.3767, Accuracy: 0.8103
Epoch 498/500 - Loss: 0.3780, Accuracy: 0.8058
Epoch 499/500 - Loss: 0.3768, Accuracy: 0.8107
Epoch 500/500 - Loss: 0.3764, Accuracy: 0.8089
Deep meta-model training complete.
Stacked (Dynamic Weighting) Ensemble Test Accuracy: 0.7842696629213484
Stacked Ensemble Classification Report:
               precision    recall  f1-score   support

           0       0.70      0.77      0.73       515
           1       0.85      0.79      0.82       820

    accuracy                           0.78      1335
   macro avg       0.77      0.78      0.78      1335
weighted avg       0.79      0.78      0.79      1335
"""