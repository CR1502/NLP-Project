############################################################################
"""
This file tests the Bagging models there is no need to re-run these files
1_data_cleaning.ipynb
2_Dataset_recreation.py
3_Bagging.ipynb
"""
############################################################################
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random

sns.set(style="whitegrid")

# Load Test Data & TF-IDF Features

test_df = pd.read_csv("Final_data/test_data.csv")
print("Test DataFrame shape:", test_df.shape)

test_texts = test_df["claim"].tolist()
y_test = test_df["label"].values

with open("Compressed model folder/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
X_test_tfidf = vectorizer.transform(test_texts)
print("TF-IDF features extracted for test set. Shape:", X_test_tfidf.shape)

# Load Saved Bagging Models

with open("Compressed model folder/final_bagging_models.pkl", "rb") as f:
    bagging_models = pickle.load(f)
print("Loaded Bagging models:", list(bagging_models.keys()))

# Evaluate Individual Bagging Models and Build Predictions Dictionary

def majority_vote_1d(row):
    # Flatten row (convert to int) and return majority vote.
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

# Unweighted Ensemble Evaluation -> All model have equal weight
all_preds = np.vstack(list(predictions_dict.values())).T
unweighted_preds = np.array([majority_vote_1d(row) for row in all_preds])
unweighted_acc = accuracy_score(y_test, unweighted_preds)
print("Unweighted Ensemble Accuracy:", unweighted_acc)
print("Unweighted Ensemble Classification Report:\n", classification_report(y_test, unweighted_preds, zero_division=1))
model_names.append("Ensemble-Unweighted")
accuracies.append(unweighted_acc)

# Leave-One-Out Analysis -> Check the accuracy of the model after leaving each one of the models in the bagging ensemble
#                           to check which one hurts or helps the overall ensemble more
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

output_folder = "Images"
os.makedirs(output_folder, exist_ok=True)

plt.figure(figsize=(10, 6))
models_loo = list(loo_accuracies.keys())
delta_values = [loo_accuracies[m] - full_ensemble_acc for m in models_loo]
plt.bar(models_loo, delta_values, color='coral')
plt.xlabel("Left-Out Model")
plt.ylabel("Accuracy Difference")
plt.title("Leave-One-Out Ensemble Analysis\n(Difference relative to full ensemble accuracy)")
plt.axhline(0, color='black', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "leave_one_out_analysis.png"), dpi=300)
plt.show()


#  Weighted Ensemble -> log reg has less weight as without it being there is less effect to the model overall

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


# Set weights: e.g., "logreg" gets 0.5; others get 1.0.
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

plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color=["blue", "green", "red", "purple", "orange", "gray", "teal"])
plt.xlabel("Model")
plt.ylabel("Test Accuracy")
plt.title("Test Set Accuracy Comparison (Bagging Models & Ensembles)")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("Images/ensemble_comparison.png", dpi=300)
plt.show()

# Live Testing Functionality with Multiple Random Samples

def live_test_claims(test_texts, y_true, vectorizer, models_dict, num_samples=5):
    """
    Randomly select a number of claims (num_samples) from test_texts.
    """
    indices = random.sample(range(len(test_texts)), num_samples)
    print("\n--- Live Testing Multiple Claims ---")
    for idx in indices:
        claim = test_texts[idx]
        actual = y_true[idx]
        print("\nClaim:", claim)
        print("Actual Label:", actual)

        X_live = vectorizer.transform([claim])
        sample_preds = []
        for model_name, model in models_dict.items():
            pred = model.predict(X_live)[0]
            sample_preds.append(pred)
            print(f"{model_name} prediction: {pred}")
        ensemble_prediction = majority_vote_1d(sample_preds)
        print("Unweighted Ensemble Prediction:", ensemble_prediction)

live_test_claims(test_texts, y_test, vectorizer, bagging_models, num_samples=5)

# Compute and Save Confusion Matrix for Unweighted Ensemble
cm = confusion_matrix(y_test, unweighted_preds)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix - Unweighted Ensemble")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("Images/confusion_matrix.png", dpi=300)
plt.show()

print("Testing complete. Saved graphs: 'leave_one_out_analysis.png', 'ensemble_comparison.png', 'confusion_matrix.png'")


############################################################################
# This is the result that I got after running this file
# Label 0 → False
# Label 1 → True
"""
Test DataFrame shape: (1335, 6)
TF-IDF features extracted for test set. Shape: (1335, 5000)
Loaded Bagging models: ['logreg', 'nb', 'dt', 'svc']
logreg Accuracy: 0.7251
logreg Classification Report:
               precision    recall  f1-score   support

           0       0.69      0.52      0.59       515
           1       0.74      0.85      0.79       820

    accuracy                           0.73      1335
   macro avg       0.71      0.69      0.69      1335
weighted avg       0.72      0.73      0.72      1335

nb Accuracy: 0.7341
nb Classification Report:
               precision    recall  f1-score   support

           0       0.69      0.56      0.62       515
           1       0.75      0.85      0.80       820

    accuracy                           0.73      1335
   macro avg       0.72      0.70      0.71      1335
weighted avg       0.73      0.73      0.73      1335

dt Accuracy: 0.6951
dt Classification Report:
               precision    recall  f1-score   support

           0       0.62      0.53      0.57       515
           1       0.73      0.80      0.76       820

    accuracy                           0.70      1335
   macro avg       0.68      0.66      0.67      1335
weighted avg       0.69      0.70      0.69      1335

svc Accuracy: 0.7378
svc Classification Report:
               precision    recall  f1-score   support

           0       0.68      0.59      0.64       515
           1       0.76      0.83      0.80       820

    accuracy                           0.74      1335
   macro avg       0.72      0.71      0.72      1335
weighted avg       0.73      0.74      0.73      1335

Unweighted Ensemble Accuracy: 0.7393258426966293
Unweighted Ensemble Classification Report:
               precision    recall  f1-score   support

           0       0.69      0.59      0.64       515
           1       0.76      0.83      0.80       820

    accuracy                           0.74      1335
   macro avg       0.73      0.71      0.72      1335
weighted avg       0.74      0.74      0.74      1335


Leave-One-Out Analysis:
Ensemble Accuracy without logreg: 0.7356 (Delta: -0.0037)
Ensemble Accuracy without nb: 0.7333 (Delta: -0.0060)
Ensemble Accuracy without dt: 0.7326 (Delta: -0.0067)
Ensemble Accuracy without svc: 0.7288 (Delta: -0.0105)
Weighted Ensemble Accuracy (logreg=0.5, others=1.0): 0.7355805243445693
Weighted Ensemble Classification Report:
               precision    recall  f1-score   support

           0       0.69      0.57      0.63       515
           1       0.76      0.84      0.80       820

    accuracy                           0.74      1335
   macro avg       0.72      0.71      0.71      1335
weighted avg       0.73      0.74      0.73      1335


--- Live Testing Multiple Claims ---

Claim: Period Pain Drug Can Cure Alzheimer’s Disease, New Study Suggests
Actual Label: 0
logreg prediction: 0
nb prediction: 0
dt prediction: 1
svc prediction: 0
Unweighted Ensemble Prediction: 0

Claim: "Dustin Diamond (aka ""Screech"") was charged with second-degree murder after stabbing a man at a bar."
Actual Label: 0
logreg prediction: 1
nb prediction: 1
dt prediction: 0
svc prediction: 1
Unweighted Ensemble Prediction: 1

Claim: In clamor to reopen, many black people feel overlooked.
Actual Label: 1
logreg prediction: 1
nb prediction: 1
dt prediction: 1
svc prediction: 1
Unweighted Ensemble Prediction: 1

Claim: “Five veterinary labs have their CLIA certification to officially test human patients. There are a lot of labs who are doing surveillance testing that don't need the CLIA certification.”
Actual Label: 1
logreg prediction: 0
nb prediction: 0
dt prediction: 0
svc prediction: 0
Unweighted Ensemble Prediction: 0

Claim: "Under current U.S. immigration policy, ""literally one person with a green card"" can, in the extreme, bring in more than 270 of his relatives."
Actual Label: 0
logreg prediction: 0
nb prediction: 0
dt prediction: 0
svc prediction: 0
Unweighted Ensemble Prediction: 0
Confusion Matrix:
 [[305 210]
 [138 682]]
Testing complete. Saved graphs: 'leave_one_out_analysis.png', 'ensemble_comparison.png', 'confusion_matrix.png'

Process finished with exit code 0
"""