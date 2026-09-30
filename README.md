# Intelligent Voting-Based Ensemble Model for Early Detection of Alzheimer's Disease

An end-to-end medical informatics and machine learning framework that predicts early-stage Alzheimer's Disease (AD) by combining structured clinical, cognitive assessments, and neuro-imaging/biological biomarker features.

Features an **Adaptive F1-Weighted Multi-Model Soft Voting Ensemble** supporting both:
1. **Standard Clinical Ensemble (4 Core Models)**: Random Forest, XGBoost, SVC, and MLP.
2. **Advanced SOTA Super-Ensemble (10 Models)**: XGBoost, CatBoost, LightGBM, Gradient Boosting, Random Forest, Extra Trees, Deep MLP, SVC, KNN, and ElasticNet Logistic Regression.

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

#### Option A: Launch the Interactive Web UI
Start the interactive diagnostic web application:
```bash
python app.py
```
Then open **`http://localhost:5000`** in your browser.

The Web UI features:
- **Interactive Engine Switcher**: Toggle between the **Advanced 10-Model Super-Ensemble** (99% Acc, 0 FP) and the **Standard 4-Model Clinical Ensemble** (96% Acc).
- **Quick Demo Presets**: Pre-populate patient metrics for *Healthy Control*, *Early MCI*, or *Probable AD*.
- **Dynamic Multi-Classifier Breakdown**: Real-time posterior probabilities $P(AD)$ and validation weights across model families.
- **Patient-Specific Local SHAP Attribution**: Instant TreeExplainer waterfall feature breakdown displaying individual risk-elevating and protective factors with clinical insights.
- **Clinical Staging & 3-Year Prognosis Trajectory**: Stratifies patients into clinical stages (*Cognitively Normal*, *Subjective Cognitive Decline*, *Mild Cognitive Impairment*, *Mild AD*, *Severe AD*) and projects risk over Months 0, 12, 24, 36.
- **Interactive 'What-If' Therapeutic Simulator**: Simulates occupational therapy, lifestyle changes, and anti-amyloid treatments, calculating absolute and relative risk reductions.
- **Formal Clinical Diagnostic Report (Print / PDF)**: Formatted medical document with patient metrics vs. normative ranges, 10-model consensus, and clinician sign-off line.
- **Cohort Batch Screening & Hospital Triage Dashboard**: Drag-and-drop CSV uploader, 1-click 25-patient demo cohort loader, triage KPI cards, filter tabs, and downloadable annotated CSV.
- **Model Evaluation Benchmarks Tab**: Comprehensive performance metrics and side-by-side confusion matrix validation ($N=200$).
- **Explainable AI (SHAP) Tab**: Publication-grade feature importance and multi-model ROC curves.

#### Option B: Run the Command-Line Pipeline & Training
Execute the full training, ensembling, and evaluation pipeline:
```bash
python main.py
```

---

## 🌟 Clinical Decision Support System (CDSS) Innovations

### 1. Patient-Specific Local SHAP Feature Attribution
Unlike global feature plots that explain the dataset as a whole, the local SHAP TreeExplainer generates an individualized attribution profile for every evaluated patient. It computes exact directional Shapley contributions ($\phi_i$), classifying whether each biomarker elevates risk (positive impact meter) or reflects preservation (protective meter), accompanied by automated clinical insights.

### 2. Clinical Disease Staging & 3-Year Prognosis Trajectory
Patients are categorized into 5 clinical stages based on ensemble risk and biomarker thresholds:
1. **Stage 1: Cognitively Normal (CN)** ($P < 0.20$)
2. **Stage 2: Subjective Cognitive Decline (SCD)** ($0.20 \le P < 0.40$)
3. **Stage 3: Mild Cognitive Impairment (Prodromal AD / MCI)** ($0.40 \le P < 0.70$)
4. **Stage 4: Mild-to-Moderate Alzheimer's Disease** ($0.70 \le P < 0.90$)
5. **Stage 5: Moderate-to-Severe Alzheimer's Disease** ($P \ge 0.90$)

A 3-year decline trajectory model projects natural progression across Month 0, Month 12, Month 24, and Month 36 to help clinicians determine the therapeutic window.

### 3. Interactive 'What-If' Intervention Simulator
Enables clinicians to test counterfactual intervention scenarios in real time:
- Occupational & functional independence therapy ($\Delta \text{Functional} \in [0, +2.5]$)
- Lifestyle & daily living skills rehabilitation ($\Delta \text{ADL} \in [0, +2.5]$)
- Anti-amyloid clearance & solubilization ($\Delta \text{CSF A}\beta_{42} \in [0, +200 \text{ pg/mL}]$)
- Cholinesterase inhibitor / cognitive training ($\Delta \text{MMSE} \in [0, +3.0]$)
Outputs absolute risk reduction ($\Delta P$) and relative risk improvement percentage.

### 4. High-Throughput Cohort Batch Screening & Triage Dashboard
Designed for hospital admission triage and clinical trial patient intake:
- Screen entire cohorts via CSV upload or instant 25-patient demo cohort loader.
- Automatically assigns triage priority: **High Priority** ($P \ge 0.65$), **Borderline** ($0.35 \le P < 0.65$), or **Normative** ($P < 0.35$).
- Live search by Patient ID, categorical filter tabs, and one-click export of annotated triage CSVs.

### 5. Exportable Clinical Diagnostic Report (Print / PDF)
Generates an institutional clinical document featuring:
- Official reference code, timestamp, and active ensemble engine.
- Patient biomarker values vs. laboratory normative reference ranges.
- Full 10-model consensus probability matrix.
- Primary local risk drivers from TreeSHAP.
- Evidence-based clinical protocol recommendations and clinician signature block.
- `@media print` optimized styling for physical printing or instant PDF export.

---

## 📊 Comprehensive 10-Model Empirical Benchmarks ($N=200$ Test Cohort)

| Model Architecture | Family | Accuracy | Precision | Recall (Sens.) | F1-Score | ROC-AUC | Validation Weight |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | Gradient Boosting | **0.9950** | 1.0000 | 0.9853 | 0.9926 | 0.9993 | $w = 0.1070$ |
| **CatBoost Classifier** | Symmetric Trees | **0.9900** | 0.9853 | 0.9853 | 0.9853 | **0.9997** | $w = 0.1062$ |
| **LightGBM Classifier** | Histogram Boosting | **0.9900** | 0.9853 | 0.9853 | 0.9853 | 0.9996 | $w = 0.1062$ |
| **Gradient Boosting** | Additive Deviance | **0.9900** | 0.9853 | 0.9853 | 0.9853 | 0.9968 | $w = 0.1062$ |
| **Random Forest** | Bagged Decision Trees | **0.9800** | 1.0000 | 0.9412 | 0.9697 | 0.9974 | $w = 0.1045$ |
| **Extra Trees** | Extremely Randomized | **0.9250** | 0.9077 | 0.8676 | 0.8872 | 0.9875 | $w = 0.0964$ |
| **Multi-Layer Perceptron (MLP)** | Deep Neural Network | **0.9300** | 0.8857 | 0.9118 | 0.8986 | 0.9785 | $w = 0.0969$ |
| **Support Vector Classifier (SVC)** | RBF Kernel | **0.9250** | 0.8841 | 0.8971 | 0.8905 | 0.9761 | $w = 0.0956$ |
| **ElasticNet Logistic Regression** | L1/L2 Regularized | **0.8950** | 0.8615 | 0.8235 | 0.8421 | 0.9615 | $w = 0.0908$ |
| **K-Nearest Neighbors (KNN)** | Instance Distance | **0.8900** | 0.8485 | 0.8235 | 0.8358 | 0.9622 | $w = 0.0901$ |
| ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── |
| 🩺 **Standard Ensemble (4 Models)** | Clinical Baseline | **0.9600** | **0.9286** | **0.9559** | **0.9420** | **0.9977** | $\sum w = 1.0$ |
| ⚡ **Advanced Super-Ensemble (10 Models)** | SOTA Multi-Model | **0.9900** | **1.0000** | **0.9706** | **0.9851** | **0.9993** | $\sum w = 1.0$ |

---

## 🎯 Confusion Matrix Validation Comparison ($N=200$ Test Set)

### Standard 4-Model Clinical Ensemble
```
                  Predicted Non-AD (0)    Predicted AD (1)
Actual Non-AD (0)          127                   5       (Specificity = 96.2%)
Actual AD (1)                3                  65       (Sensitivity = 95.6%)
```
- Accuracy: **96.0%**
- False Positives: **5** | False Negatives: **3**

### Advanced 10-Model SOTA Super-Ensemble
```
                  Predicted Non-AD (0)    Predicted AD (1)
Actual Non-AD (0)          132                   0       (Specificity = 100.0% - ZERO False Positives!)
Actual AD (1)                2                  66       (Sensitivity = 97.1%)
```
- Accuracy: **99.0%** | Precision: **100.0%**
- False Positives: **0** (Zero False Alarms) | False Negatives: **2**

---

## 🔬 Features & Architecture

### Feature Set (11 Features)
- **Cognitive & Clinical**: `MMSE`, `MemoryComplaints`, `FunctionalAssessment`, `Confusion`, `ADL`, `BehavioralProblems`
- **Biomarkers & Neuro-imaging**: `Hippocampal_Volume`, `MRI_PET_Imaging_Scores`, `CSF Abeta42 Levels`, `APOE4 Gene Presence`, `Age`
- **Diagnostic Target**: `Diagnosis` (`0 = Non-Alzheimer`, `1 = Alzheimer`)

### Adaptive F1-Weighted Soft-Voting Aggregation
Individual models predict posterior class probabilities $P_i(y=1 \mid \mathbf{x})$. The ensemble aggregates them dynamically weighted by their cross-validated validation F1-scores:
$$w_i = \text{F1}_i(y_{\text{val}}, \hat{y}_i)$$
$$P_{\text{ensemble}}(y=1 \mid \mathbf{x}) = \frac{\sum_{i=1}^{M} w_i \cdot P_i(y=1 \mid \mathbf{x})}{\sum_{i=1}^{M} w_i}$$
Decision rule:
$$\hat{y}_{\text{ensemble}} = \mathbb{I}\left(P_{\text{ensemble}} \ge 0.50\right)$$

---

## 📂 Project Structure & Artifacts

| File | Description |
| :--- | :--- |
| `app.py` | Production Flask web application backend supporting both Standard (4M) and Advanced (10M) modes. |
| `main.py` | Complete, self-contained end-to-end training, ensembling, evaluation, and explainability pipeline. |
| `generate_datasets.py` | Data preparation script generating `alzheimers_disease_data.csv` and `alz.csv`. |
| `generate_advanced_visualizations.py` | Generates publication-grade evaluation charts for all 10 models + both ensembles. |
| `alzheimers_disease_data.csv` | Primary clinical and cognitive assessment cohort ($N=1000$). |
| `alz.csv` | Extracted neuro-imaging and biological biomarkers ($N=1000$). |
| `models_bundle_advanced.pkl` | Serialized 10-model bundle with scalers, weights, and benchmarks. |
| `models_bundle.pkl` | Serialized 4-model core clinical bundle. |
| `best_model_xgb.pkl` | Serialized standalone XGBoost classifier artifact. |
| `scaler.pkl` | Serialized `StandardScaler` feature transformer. |
| `model_accuracy_comparison.png` | Bar chart comparing accuracy and F1 scores across 10 models + both ensembles. |
| `confusion_matrix.png` | Side-by-side confusion matrix comparison (Standard 96% vs Super-Ensemble 99%). |
| `roc_curve.png` | Multi-model ROC-AUC curves comparing all major classifiers and ensembles. |
| `shap_summary.png` | SHAP beeswarm plot displaying feature importance hierarchy. |
| `templates/index.html` | Modern medical glassmorphism Web UI template with mode toggle and presets. |
| `static/css/style.css` | Comprehensive stylesheet with dark/light themes and responsive design. |
| `static/js/app.js` | Interactive client application with dynamic multi-model progress meters. |
| `requirements.txt` | Python library dependencies including `xgboost`, `lightgbm`, and `catboost`. |
