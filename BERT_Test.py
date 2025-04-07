import torch
from transformers import BertForSequenceClassification, BertTokenizerFast
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ----------------------------
# 1. Load Test Data
# ----------------------------
test_df = pd.read_csv("Final_data/test_data.csv")
print("Test DataFrame shape:", test_df.shape)

# Assume the test file has a 'claim' column (text) and a 'label' column (numeric labels)
test_texts = test_df["claim"].tolist()
y_test = test_df["label"].values

# ----------------------------
# 2. Load the Saved Final BERT Model and Tokenizer
# ----------------------------
tokenizer = BertTokenizerFast.from_pretrained("Bert_Model_Final_Tokenizer")
model = BertForSequenceClassification.from_pretrained("Bert_Model_Final")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"BERT model loaded on {device}!")

# ----------------------------
# 3. Tokenize the Test Data
# ----------------------------
def tokenize_texts(texts, tokenizer, max_length=128):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

test_encodings = tokenize_texts(test_texts, tokenizer, max_length=128)
input_ids = test_encodings["input_ids"].to(device)
attention_mask = test_encodings["attention_mask"].to(device)

# ----------------------------
# 4. Run Predictions and Evaluate
# ----------------------------
model.eval()
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).cpu().numpy()

acc = accuracy_score(y_test, preds)
print("Test Accuracy:", acc)
print("Classification Report:")
print(classification_report(y_test, preds, zero_division=1))

# ----------------------------
# 5. Visualize Test Accuracy (Horizontal Bar Chart)
# ----------------------------
plt.figure(figsize=(8,6))
plt.barh(["Final BERT Model"], [acc], color=["orange"])
plt.xlabel("Test Accuracy")
plt.title("Final BERT Model Test Accuracy")
plt.xlim(0, 1)
plt.tight_layout()
plt.show()