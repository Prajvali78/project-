# Intelligent Voting-Based Ensemble Model for Early Detection of Alzheimer's Disease

An end-to-end machine learning framework that predicts early-stage Alzheimer's Disease (AD) by combining structured clinical, cognitive assessments, and neuro-imaging/biological biomarker features using an **adaptive F1-weighted soft-voting ensemble**.

---

## 🚀 Quick Start Guide (How to Run After Cloning)

### 1. Clone the Repository
Open your terminal (PowerShell, Command Prompt, or Bash) and run:
```bash
git clone https://github.com/Prajvali78/project-.git
cd project-
```

### 2. Create and Activate a Virtual Environment
It is recommended to use a virtual environment to manage dependencies:

- **Windows (PowerShell / Command Prompt)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```

- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
Install all required packages:
```bash
pip install -r requirements.txt
```

### 4. Run the Project
Execute the main pipeline:
```bash
python main.py
```

---

## 📂 Project Structure & Artifacts

| File | Description |
| :--- | :--- |
| `main.py` | Complete, self-contained end-to-end training, ensembling, evaluation, and explainability pipeline. |
| `generate_datasets.py` | Data preparation script generating `alzheimers_disease_data.csv` and `alz.csv`. |
| `alzheimers_disease_data.csv` | Primary clinical and cognitive assessment cohort ($N=1000$). |
| `alz.csv` | Extracted neuro-imaging and biological biomarkers ($N=1000$). |
| `best_model_xgb.pkl` | Serialized primary XGBoost classifier artifact. |
| `scaler.pkl` | Serialized `StandardScaler` feature transformer. |
| `model_accuracy_comparison.png` | Bar chart comparing accuracy and F1 scores of all models. |
| `confusion_matrix.png` | Test set confusion matrix ($N=200$, $TN=127, FP=5, FN=3, TP=65$). |
| `roc_curve.png` | Multi-classifier ROC-AUC curves ($AUC \approx 0.998$). |
| `shap_summary.png` | SHAP beeswarm plot displaying feature importance hierarchy. |
| `requirements.txt` | Python library dependencies. |

---

## 🔬 Features & Architecture

### Feature Set (11 Features)
- **Cognitive & Clinical**: `MMSE`, `MemoryComplaints`, `FunctionalAssessment`, `Confusion`, `ADL`, `BehavioralProblems`
- **Biomarkers & Neuro-imaging**: `Hippocampal_Volume`, `MRI_PET_Imaging_Scores`, `CSF Abeta42 Levels`, `APOE4 Gene Presence`, `Age`
- **Target**: `Diagnosis` (`0 = Non-Alzheimer`, `1 = Alzheimer`)

### Base Classifiers & Ensemble
- **Random Forest**: 300 estimators
- **XGBoost**: 300 estimators, learning rate = 0.05, max depth = 6
- **Support Vector Classifier (SVC)**: RBF kernel, probability = True
- **Multi-Layer Perceptron (MLP)**: hidden layers = (128, 64), max iter = 500
- **Adaptive Soft-Voting**: Dynamically weights models based on their validation F1-scores:
  $$P_{\text{ensemble}} = \frac{\sum w_i \cdot P_i}{\sum w_i}$$
