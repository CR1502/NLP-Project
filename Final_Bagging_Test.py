import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizerFast
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

test_df = pd.read_csv("Final_data/test_data.csv")
print("Test DataFrame shape:", test_df.shape)
test_texts = test_df["claim"].tolist()
y_test = test_df["label"].values

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
X_test_tfidf = vectorizer.transform(test_texts)
print("TF-IDF features extracted for test set. Shape:", X_test_tfidf.shape)

with open("final_bagging_models.pkl", "rb") as f:
    bagging_models = pickle.load(f)
print("Loaded Bagging models:", list(bagging_models.keys()))

def majority_vote_1d(row):
    return np.bincount(np.asarray(row).flatten().astype(np.int64)).argmax()

model_names = []
accuracies = []
predictions_dict = {}

for model_name, model in bagging_models.items():
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    model_names.append(model_name)
    accuracies.append(acc)
    predictions_dict[model_name] = y_pred
    print(f"{model_name} Accuracy: {acc:.4f}")
    print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))

all_preds = np.vstack(list(predictions_dict.values())).T
unweighted_preds = np.array([majority_vote_1d(row) for row in all_preds])
unweighted_acc = accuracy_score(y_test, unweighted_preds)
print("Unweighted Ensemble Accuracy:", unweighted_acc)
print("Unweighted Ensemble Classification Report:\n", classification_report(y_test, unweighted_preds, zero_division=1))
model_names.append("Ensemble-Unweighted")
accuracies.append(unweighted_acc)

print("\nLeave-One-Out Analysis:")
full_ensemble_acc = unweighted_acc
loo_accuracies = {}
for key in predictions_dict:
    remaining_models = [pred for k, pred in predictions_dict.items() if k != key]
    remaining_preds = np.vstack(remaining_models).T
    loo_preds = np.array([majority_vote_1d(row) for row in remaining_preds])
    loo_acc = accuracy_score(y_test, loo_preds)
    loo_accuracies[key] = loo_acc
    print(f"Ensemble Accuracy without {key}: {loo_acc:.4f} (Delta: {loo_acc - full_ensemble_acc:+.4f})")

plt.figure(figsize=(10,6))
models_loo = list(loo_accuracies.keys())
delta_values = [loo_accuracies[m] - full_ensemble_acc for m in models_loo]
plt.bar(models_loo, delta_values, color='coral')
plt.xlabel("Left-Out Model")
plt.ylabel("Accuracy Difference")
plt.title("Leave-One-Out Ensemble Analysis\n(Difference relative to full ensemble accuracy)")
plt.axhline(0, color='black', linestyle='--')
plt.tight_layout()
plt.savefig("leave_one_out_analysis.png", dpi=300)
plt.show()

model_order = sorted(predictions_dict.keys())
all_preds_ordered = np.vstack([predictions_dict[name] for name in model_order]).T

def weighted_majority_vote(preds, weights):
    n_samples, n_models = preds.shape
    ensemble_preds = []
    for i in range(n_samples):
        vote_dict = {}
        for j in range(n_models):
            label = int(preds[i, j])
            vote_dict[label] = vote_dict.get(label, 0) + weights[j]
        ensemble_preds.append(max(vote_dict, key=vote_dict.get))
    return np.array(ensemble_preds)

weights = []
for name in model_order:
    if name == "logreg":
        weights.append(0.5)
    else:
        weights.append(1.0)

weighted_preds = weighted_majority_vote(all_preds_ordered, weights)
weighted_acc = accuracy_score(y_test, weighted_preds)
print("Weighted Ensemble Accuracy (logreg=0.5, others=1.0):", weighted_acc)
print("Weighted Ensemble Classification Report:\n", classification_report(y_test, weighted_preds, zero_division=1))
model_names.append("Ensemble-Weighted")
accuracies.append(weighted_acc)

plt.figure(figsize=(10,6))
plt.bar(model_names, accuracies, color=["blue", "green", "red", "purple", "orange", "gray", "teal"])
plt.xlabel("Model")
plt.ylabel("Test Accuracy")
plt.title("Test Set Accuracy Comparison (Bagging Models & Ensembles)")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("ensemble_comparison.png", dpi=300)
plt.show()