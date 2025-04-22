# MediScan: A Hybrid Ensemble Approach to Detecting Misinformation in Medical Claims

This repository implements an ensemble-based NLP pipeline combining bagging classifiers and a BERT-based transformer under a meta-learning framework to detect false medical claims using the PUBHEALTH dataset.

---

## 📁 Repository Structure

```plaintext
├── 1_data_cleaning.ipynb           # Combine & clean PUBHEALTH TSV files
├── 2.1_data_vis.py                 # Visualizations for class distribution
├── 2.2_data_vis.py                 # Visualizations for claim length distribution
├── 2_Dataset_recreation.py         # Split and recreate the cleaned dataset
├── 3_Bagging.ipynb                 # Train individual bagging classifiers
├── 4_Bagging_Test.py               # Evaluate bagging models on test set
├── 5_BERT.ipynb                    # Fine-tune BERT and evaluate on dev
├── 6_BERT_Test.py                  # Load BERT model and test
├── 7_final_ensemble.py             # Stack BERT + bagging + meta-learner
├── 8_Downloading_required_files    # Download models, CSVs, tokenizers from Drive
├── README.md                       # This file
```

## 📦 Required Dependencies
```plaintext
numpy==1.24.4
pandas==1.5.3
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
nltk==3.8.1
transformers==4.31.0
torch==2.0.1
tqdm==4.65.0
tokenizers==0.13.3
ipykernel==6.22.0
notebook==6.5.4
joblib==1.2.0
regex==2023.3.23
scipy==1.10.1
```

## 🛠️ Environment Setup Instructions
```plaintext
MacOS
  1. Clone the repository:
  git clone https://github.com/your-repo/mediscan.git
  cd mediscan
  2. Create and activate a virtual environment:
  python3 -m venv venv
  source venv/bin/activate
 	3. Install dependencies:
  pip install -r requirements.txt
  4. Download all required files from Google Drive (from the Downloading required files.txt) and place them in the working directory.
```
```plaintext
Windows
  1. Clone the repository:
  git clone https://github.com/your-repo/mediscan.git
  cd mediscan
  2. Create and activate a virtual environment:
  python -m venv venv
  venv\Scripts\activate
  3. Install dependencies:
  pip install -r requirements.txt
  4. Download all required files from Google Drive (from the Downloading required files.txt) and place them in the working directory.
```

```plaintext
The folder in the Google Drive contains these files:

	•	BERT model weights (Bert_Model_Final, Bert_Model_Final_Tokenizer)
	•	CSV files used for evaluation
	•	All prediction outputs from bert_output, bert_output_final, etc.
	•	Intermediate data (Final_data, Compressed model folder)
  	•	Images generated from all the files.
```
## 🚀 Running the Code
```plaintext
You do NOT need to run all files from scratch. Only the final ensemble script needs to be executed.

	1. Make sure all data folders and model checkpoints from the Google Drive are placed in your root working directory.
	2. Then run: python 7_final_ensemble.py
```
## 🧪 Testing
```plaintext
To re-test the BERT model or the bagging models individually:
python 4_Bagging_Test.py
python 6_BERT_Test.py
python 7_final_ensemble.py

To rerun the full pipeline (optional):
Run all scripts in order
jupyter notebook 1_data_cleaning.ipynb
python 2_Dataset_recreation.py
jupyter notebook 3_Bagging.ipynb
python 4_Bagging_Test.py
jupyter notebook 5_BERT.ipynb
python 6_BERT_Test.py
python 7_final_ensemble.py
```

## 🧠 Authors
	• Craig Lionel Roberts
	• Priyanshu Srivastava















