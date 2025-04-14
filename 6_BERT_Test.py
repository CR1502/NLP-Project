############################################################################
"""
This file tests the BERT there is no need to re-run these files
1_data_cleaning.ipynb
2_Dataset_recreation.py
3_Bagging.ipynb
4_Bagging_Test.py
5_BERT.ipynb
"""
############################################################################
import pandas as pd
import numpy as np
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Set visualization style
sns.set(style="whitegrid")

# Load Test Data
test_df = pd.read_csv("Final_data/test_data.csv")
print("Test DataFrame shape:", test_df.shape)
# Extract claim texts and true labels
test_texts = test_df["claim"].tolist()
y_test = test_df["label"].values

# Load the Saved Final BERT Model and Tokenizer
tokenizer = BertTokenizerFast.from_pretrained("Bert_Model_Final_Tokenizer")
model = BertForSequenceClassification.from_pretrained("Bert_Model_Final")
device = torch.device("cpu")
model.to(device)
print(f"Final BERT model loaded on {device}!")

# Tokenize the Test Data
def tokenize_texts(texts, tokenizer, max_length=128):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

test_encodings = tokenize_texts(test_texts, tokenizer, max_length=128)
input_ids = test_encodings["input_ids"].to(device)
attention_mask = test_encodings["attention_mask"].to(device)

# Run Predictions and Evaluate on Test Set
model.eval()
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).cpu().numpy()

acc = accuracy_score(y_test, preds)
print("Test Accuracy:", acc)
print("Classification Report:")
print(classification_report(y_test, preds, zero_division=1))

# Plot and Save the Confusion Matrix
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix - Final BERT Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("Images/confusion_matrix_bert.png", dpi=300)
plt.show()

# Visualize Test Accuracy
plt.figure(figsize=(8, 6))
plt.barh(["Final BERT Model"], [acc], color=["orange"])
plt.xlabel("Test Accuracy")
plt.title("Final BERT Model Test Accuracy")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig("Images/final_bert_test_accuracy.png", dpi=300)
plt.show()

# Visualize Confidence Scores Histogram
probs = torch.softmax(logits, dim=1).cpu().numpy()
confidence_scores = np.max(probs, axis=1)

plt.figure(figsize=(8, 6))
sns.histplot(confidence_scores, bins=20, kde=True, color="purple")
plt.xlabel("Confidence Score")
plt.ylabel("Number of Test Samples")
plt.title("Distribution of Confidence Scores for Test Predictions")
plt.tight_layout()
plt.savefig("Images/confidence_histogram_bert.png", dpi=300)
plt.show()


# Live Testing Functionality with Multiple Random Samples
def live_test_claims_random(test_texts, y_true, tokenizer, model, device, num_samples=5):
    """
    Randomly select num_samples from test_texts.
    """
    indices = random.sample(range(len(test_texts)), num_samples)
    print("\n--- Live Testing Samples ---")
    for idx in indices:
        claim = test_texts[idx]
        actual_label = y_true[idx]
        encoding = tokenize_texts([claim], tokenizer, max_length=128)
        input_ids_live = encoding["input_ids"].to(device)
        attention_mask_live = encoding["attention_mask"].to(device)
        model.eval()
        with torch.no_grad():
            output = model(input_ids_live, attention_mask=attention_mask_live)
            sample_probs = torch.softmax(output.logits, dim=1).cpu().numpy()[0]
            pred_label = int(np.argmax(sample_probs))
            confidence = float(np.max(sample_probs))
        print("\nClaim:", claim)
        print("Actual Label:", actual_label)
        print("Predicted Label:", pred_label, "with confidence:", f"{confidence:.4f}")


live_test_claims_random(test_texts, y_test, tokenizer, model, device, num_samples=5)

print("BERT testing complete. Saved graphs: 'confusion_matrix_bert.png', 'final_bert_test_accuracy.png', 'confidence_histogram_bert.png'")


############################################################################
# This is the result that I got after running this file
"""
Test DataFrame shape: (1335, 6)
Final BERT model loaded on cpu!
Test Accuracy: 0.7835205992509363
Classification Report:
              precision    recall  f1-score   support

           0       0.70      0.78      0.73       515
           1       0.85      0.79      0.82       820

    accuracy                           0.78      1335
   macro avg       0.77      0.78      0.78      1335
weighted avg       0.79      0.78      0.79      1335


--- Live Testing Samples ---

Claim: Biosynthetic Corneas Show Promise in Transplants
Actual Label: 1
Predicted Label: 1 with confidence: 0.5941

Claim: A patient discharge document proves that “The WHO and CDC do NOT recommend that healthy people wear masks.”
Actual Label: 0
Predicted Label: 0 with confidence: 0.8147

Claim: In 2010, Clinton "said Iran could enrich uranium. In 2014 she said she’s always argued against it.
Actual Label: 0
Predicted Label: 0 with confidence: 0.6025

Claim: Doctors’ long-running advice: Get checked before a marathon.
Actual Label: 1
Predicted Label: 1 with confidence: 0.9986

Claim: On a bipartisan task force on ways to improve fiscal policy.
Actual Label: 0
Predicted Label: 1 with confidence: 0.6620

BERT testing complete. Saved graphs: 'confusion_matrix_bert.png', 'final_bert_test_accuracy.png', 'confidence_histogram_bert.png'

Process finished with exit code 0
"""