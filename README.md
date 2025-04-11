# MediScan NLP Project

This project implements an NLP-based system for detecting fake medical news and misinformation using an ensemble approach. The pipeline consists of several stages:

- **Data Cleaning:** Preprocess and clean medical claims and related text from the PUBHEALTH dataset.
- **Bagging Models:** Train multiple bagging ensemble models (e.g., logistic regression, Naive Bayes, decision trees, SVC) using TF-IDF features.
- **BERT Transfer Model:** Fine-tune a BERT-based model for sequence classification.
- **Ensemble Stacking:** Combine the predictions from the bagging ensemble and the BERT model using a dynamic, instance-based meta-model (Deep Meta-Ensemble).


## File Structure

- **Final_data_cleaning.ipynb**  
  Contains code to load, clean, and prepare the PUBHEALTH dataset. The cleaned data is saved as `final_cleaned_file.csv`.

- **data_splitter.py**  
  A script that splits `final_cleaned_file.csv` into training (60%), development (25%), and test sets (15%). These splits are saved in the `Final_data` folder as `train_data.csv`, `dev_data.csv`, and `test_data.csv`.

- **bagging_models.ipynb**  
  Notebook that:
  - Extracts TF-IDF features using a pre-saved vectorizer.
  - Trains individual bagging models (logistic regression, Naive Bayes, decision tree, and SVC).
  - Evaluates each model on the development set and saves the ensemble in `final_bagging_models.pkl`.

- **bert_model_training.ipynb**  
  Notebook that:
  - Loads and tokenizes the training and development datasets.
  - Fine-tunes a pre-trained BERT model on the training data with hyperparameter grid search.
  - Saves the final BERT model and tokenizer to `Bert_Model_Final` and `Bert_Model_Final_Tokenizer`, respectively, and pickles the state dictionary.

- **final_ensemble.ipynb**  
  Notebook that:
  - Loads the test dataset (from `Final_data/test_data.csv`).
  - Loads the pre-saved TF-IDF vectorizer, bagging models, and final BERT model.
  - Computes probability predictions from the bagging ensemble and BERT.
  - Constructs meta-features (including a normalized claim length feature).
  - Loads and trains a deep meta-model (an MLP) using a dynamic weighting strategy.
  - Evaluates the stacked (dynamic weighting) ensemble on the test set and visualizes & saves the resulting accuracy graph.

## Requirements

- Python 3.8+
- PyTorch  
- Transformers  
- scikit-learn  
- pandas, numpy  
- matplotlib, seaborn  
- datasets  
- nltk  
- langdetect

Install required packages (assuming you use `pip`):

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn datasets nltk langdetect
