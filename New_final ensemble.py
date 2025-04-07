import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, BertTokenizerFast
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

########################################
# Section 1: Load Dev & Test Data and Compute Additional Feature (Claim Length)
########################################
# Load dev and test data
dev_df = pd.read_csv("Final_data/dev_data.csv")
test_df = pd.read_csv("Final_data/test_data.csv")
print("Dev DataFrame shape:", dev_df.shape)
print("Test DataFrame shape:", test_df.shape)

# Extract claim texts and labels
dev_texts = dev_df["claim"].tolist()
y_dev = dev_df["label"].values
test_texts = test_df["claim"].tolist()
y_test = test_df["label"].values

# Compute claim length (number of words) and normalize by maximum length from dev set
def compute_length(text):
    return len(text.split())

dev_lengths = np.array([compute_length(txt) for txt in dev_texts])
test_lengths = np.array([compute_length(txt) for txt in test_texts])
max_length_dev = dev_lengths.max()
dev_lengths_norm = (dev_lengths / max_length_dev).reshape(-1, 1)
test_lengths_norm = (test_lengths / max_length_dev).reshape(-1, 1)

########################################
# Section 2: Load TF-IDF Vectorizer and Bagging Model Predictions
########################################
# Load the TF-IDF vectorizer (used by bagging models)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
X_dev_tfidf = vectorizer.transform(dev_texts)
X_test_tfidf = vectorizer.transform(test_texts)
print("TF-IDF features shape (dev):", X_dev_tfidf.shape)
print("TF-IDF features shape (test):", X_test_tfidf.shape)

# Load saved bagging models (with kNN removed)
with open("final_bagging_models.pkl", "rb") as f:
    bagging_models = pickle.load(f)
print("Loaded Bagging models:", list(bagging_models.keys()))

# Function to compute averaged probability predictions from bagging ensemble
def get_bagging_probs(models, X):
    probs_list = []
    for name, model_bag in models.items():
        p = model_bag.predict_proba(X)  # (n_samples, n_classes)
        probs_list.append(p)
    return np.mean(np.stack(probs_list, axis=0), axis=0)

bagging_probs_dev = get_bagging_probs(bagging_models, X_dev_tfidf)
bagging_probs_test = get_bagging_probs(bagging_models, X_test_tfidf)

########################################
# Section 3: Load Final BERT Model and Compute Predictions
########################################
# Load final BERT model and tokenizer from saved directories
tokenizer_final = BertTokenizerFast.from_pretrained("Bert_Model_Final_Tokenizer")
model_final = BertForSequenceClassification.from_pretrained("Bert_Model_Final")
model_final.to(device)
print(f"Final BERT model loaded on {device}!")

# Helper function to tokenize texts for BERT
def tokenize_texts(texts, tokenizer, max_length=128):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

# Function to obtain probability predictions from BERT
def get_bert_probs(texts, tokenizer, model, device, max_length=128):
    encodings = tokenize_texts(texts, tokenizer, max_length)
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (n_samples, n_classes)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return probs

bert_probs_dev = get_bert_probs(dev_texts, tokenizer_final, model_final, device, max_length=128)
bert_probs_test = get_bert_probs(test_texts, tokenizer_final, model_final, device, max_length=128)

########################################
# Section 4: Build Meta-Features for Stacking (Include Normalized Claim Length)
########################################
# For binary classification, BERT and bagging each yield 2 probabilities.
# Concatenate: [BERT_probs (2), Bagging_probs (2), normalized claim length (1)]
# -> Meta-feature vector shape = (n_samples, 5)
X_meta_dev = np.hstack([bert_probs_dev, bagging_probs_dev, dev_lengths_norm])
X_meta_test = np.hstack([bert_probs_test, bagging_probs_test, test_lengths_norm])
print("Meta-feature shape (dev):", X_meta_dev.shape)
print("Meta-feature shape (test):", X_meta_test.shape)

########################################
# Section 5: Define Meta-Dataset and a Deeper Meta-Model (MLP)
########################################
# Define a PyTorch Dataset for the meta-features and labels
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

# Define a deeper meta-model with four hidden layers and dropout.
class DeepMetaEnsemble(nn.Module):
    def __init__(self, input_dim=5, hidden_dims=[64, 32, 16, 8], dropout_rate=0.3):
        super(DeepMetaEnsemble, self).__init__()
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

        self.fc5 = nn.Linear(hidden_dims[3], 2)  # Output two weights for BERT and Bagging

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.relu4(x)
        x = self.dropout4(x)

        x = self.fc5(x)
        weights = torch.softmax(x, dim=1)
        return weights

meta_model = DeepMetaEnsemble(input_dim=5, hidden_dims=[64, 32, 16, 8], dropout_rate=0.3).to(device)
print("Deep meta-model initialized.")

########################################
# Section 6: Train the Deep Meta-Model with Increased Epochs and Tuned Learning Rate
########################################
from torch.optim import Adam

optimizer = Adam(meta_model.parameters(), lr=5e-4)
loss_fn = nn.NLLLoss()
meta_train_loader = DataLoader(meta_dev_dataset, batch_size=32, shuffle=True)

num_epochs = 50
meta_model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0
    for batch in meta_train_loader:
        features = batch["features"].to(device)  # shape: (batch_size, 5)
        labels = batch["label"].to(device)         # shape: (batch_size,)

        optimizer.zero_grad()
        # Get dynamic weights from the meta-model (shape: (batch_size, 2))
        weights = meta_model(features)

        # Extract base probabilities: first 2 columns are BERT, next 2 are bagging.
        p_bert_batch = features[:, :2]
        p_bagging_batch = features[:, 2:4]

        # Compute combined probability using dynamic instance-based weights
        combined_probs = weights[:, 0].unsqueeze(1) * p_bert_batch + weights[:, 1].unsqueeze(1) * p_bagging_batch

        # Compute log probabilities for NLLLoss (add small epsilon for numerical stability)
        log_probs = torch.log(combined_probs + 1e-8)
        loss = loss_fn(log_probs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * features.size(0)
        preds = torch.argmax(combined_probs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss /= total
    epoch_acc = correct / total
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

print("Deep meta-model training complete.")

########################################
# Section 7: Evaluate the Deep Meta-Model on Test Data and Save the Graph
########################################
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

# Visualize test accuracy of the meta-model
plt.figure(figsize=(6, 6))
plt.barh(["Stacked Ensemble (Dynamic Weighting)"], [meta_acc], color=["purple"])
plt.xlabel("Test Accuracy")
plt.title("Stacked Ensemble Test Accuracy")
plt.xlim(0, 1)
plt.tight_layout()

# Save the graph as a PNG file
plt.savefig("stacked_ensemble_accuracy.png", dpi=300)
plt.show()

torch.save(meta_model.state_dict(), "meta_model_final.pt")