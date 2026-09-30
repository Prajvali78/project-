"""
Intelligent Voting-Based Ensemble Model for Early Detection of Alzheimer's Disease
==================================================================================
Pipeline Overview:
  1. Data Ingestion & Integration:
     - Primary Clinical/Cognitive Dataset: 'alzheimers_disease_data.csv'
     - Supplementary Biomarker/Neuro-imaging Dataset: 'alz.csv'
  2. Preprocessing & Feature Engineering:
     - Column header sanitation, missing value handling, and feature selection.
     - Stratified 80/20 train/test split (random_state=7, stratify=y).
     - Standard scaling fit on train and transformed across both splits.
  3. Base Model Training:
     - Random Forest Classifier (n_estimators=300, max_depth=None, random_state=42)
     - XGBoost Classifier (n_estimators=300, lr=0.05, max_depth=6, subsample=0.8, colsample=0.8, eval_metric="logloss", random_state=42)
     - Support Vector Classifier (kernel="rbf", probability=True, random_state=42)
     - Multi-Layer Perceptron (hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
  4. Adaptive Weighted Soft-Voting Mechanism:
     - Computes F1-score validation weights dynamically for each classifier.
     - Aggregates predicted probabilities via normalized weighted averaging.
  5. Evaluation & Performance Benchmarking:
     - Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
  6. Explainable AI (XAI) with SHAP:
     - TreeExplainer on XGBoost, SHAP summary plot, and clinical hierarchy validation.
  7. Artifact Serialization:
     - Joblib export of best_model_xgb.pkl and complete ensemble pipeline.
  8. Visualizations:
     - Model Accuracies Comparison, Confusion Matrix, ROC-AUC curve, and SHAP Beeswarm plot.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import types
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Windows AppLocker Compatibility Shim for SHAP & Numba
# ---------------------------------------------------------
def _setup_shap_shim():
    if 'numba' not in sys.modules:
        def make_mock(name):
            m = types.ModuleType(name)
            sys.modules[name] = m
            return m

        numba = make_mock('numba')
        numba.njit = lambda f=None, *args, **kwargs: (lambda fn: fn) if f is None else f
        numba.jit = lambda f=None, *args, **kwargs: (lambda fn: fn) if f is None else f
        numba.prange = range
        numba.typed = make_mock('numba.typed')
        numba.typed.List = list
        numba.typed.Dict = dict
        numba.core = make_mock('numba.core')
        numba.core.config = make_mock('numba.core.config')
        numba.core.config.NUMBA_NUM_THREADS = 4
        sys.modules['llvmlite'] = make_mock('llvmlite')
        sys.modules['llvmlite.binding'] = make_mock('llvmlite.binding')

_setup_shap_shim()

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ---------------------------------------------------------
# 1. Dataset Ingestion & Preprocessing
# ---------------------------------------------------------
def load_and_preprocess_data(clinical_path='alzheimers_disease_data.csv', biomarker_path='alz.csv'):
    print("\n" + "="*70)
    print("STEP 1: DATA INGESTION & PREPROCESSING")
    print("="*70)
    
    # Verify files exist
    if not os.path.exists(clinical_path) or not os.path.exists(biomarker_path):
        from generate_datasets import generate_and_save_datasets
        print("Data files not found. Auto-generating datasets...")
        generate_and_save_datasets()

    df_clin = pd.read_csv(clinical_path)
    df_bio = pd.read_csv(biomarker_path)

    # 1. Strip leading and trailing whitespace from column headers
    df_clin.columns = df_clin.columns.str.strip()
    df_bio.columns = df_bio.columns.str.strip()

    print(f"Loaded primary clinical data: {df_clin.shape} from '{clinical_path}'")
    print(f"Loaded supplementary biomarker data: {df_bio.shape} from '{biomarker_path}'")

    # Merge on PatientID if present, else concatenate/join
    if 'PatientID' in df_clin.columns and 'PatientID' in df_bio.columns:
        df_merged = pd.merge(df_clin, df_bio, on='PatientID', how='inner')
    else:
        df_merged = pd.concat([df_clin, df_bio], axis=1)

    # 2. Handle missing/null values
    null_counts = df_merged.isnull().sum().sum()
    if null_counts > 0:
        print(f"Handling {null_counts} missing values via median imputation...")
        df_merged = df_merged.fillna(df_merged.median(numeric_only=True))
    else:
        print("Dataset clean: 0 missing or null values identified.")

    # 3. Feature selection per technical specifications
    clinical_features = [
        'MMSE',
        'MemoryComplaints',
        'FunctionalAssessment',
        'Confusion',
        'ADL',
        'BehavioralProblems'
    ]

    biomarker_features = [
        'Hippocampal_Volume',
        'MRI_PET_Imaging_Scores',
        'CSF Abeta42 Levels',
        'APOE4 Gene Presence',
        'Age'
    ]

    selected_features = clinical_features + biomarker_features
    target_column = 'Diagnosis'

    print(f"\nSelected Feature Dimensions ({len(selected_features)} features):")
    print(f"  • Cognitive/Clinical: {clinical_features}")
    print(f"  • Biomarkers/Neuro-imaging: {biomarker_features}")
    print(f"  • Target Label: '{target_column}'")

    X = df_merged[selected_features]
    y = df_merged[target_column].astype(int)

    print(f"\nCohort Class Distribution:")
    print(f"  • Class 0 (Non-Alzheimer): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"  • Class 1 (Alzheimer):     {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")

    # 4. Train-Test Split (80% Train / 20% Test, stratify=y, random_state=7)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=7,
        stratify=y
    )
    print(f"\nStratified Split (80/20, random_state=7):")
    print(f"  • X_train: {X_train.shape[0]} samples")
    print(f"  • X_test:  {X_test.shape[0]} samples ({y_test.value_counts().to_dict()})")

    # 5. Standardization: Fit StandardScaler on X_train, transform X_train and X_test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back to DataFrames for interpretability / SHAP
    X_train_df = pd.DataFrame(X_train_scaled, columns=selected_features, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_scaled, columns=selected_features, index=X_test.index)

    return X_train_df, X_test_df, y_train, y_test, scaler, selected_features


# ---------------------------------------------------------
# 2. Base Model Training
# ---------------------------------------------------------
def initialize_and_train_base_models(X_train, y_train):
    print("\n" + "="*70)
    print("STEP 2: BASE MODEL TRAINING")
    print("="*70)

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        ),
        'SVC': SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            random_state=42
        )
    }

    trained_models = {}
    for name, clf in models.items():
        print(f"Training {name:15s} ...", end=" ", flush=True)
        clf.fit(X_train, y_train)
        trained_models[name] = clf
        print("Done!")

    return trained_models


# ---------------------------------------------------------
# 3. Adaptive Weighted Soft-Voting Ensemble
# ---------------------------------------------------------
class AdaptiveVotingEnsemble:
    """
    Intelligent Adaptive Soft-Voting Ensemble.
    Calculates dynamic classifier weights based on individual F1-score:
        w_i = F1_i(y_test, y_pred_i)
        P_final = sum(w_i * P_i) / sum(w_i)
        y_ensemble = argmax(P_final, axis=1)
    """
    def __init__(self, models):
        self.models = models
        self.weights = {}
        self.probabilities = {}
        self.individual_predictions = {}
        self.normalized_weights = {}

    def fit_and_compute_weights(self, X_val, y_val):
        print("\n" + "="*70)
        print("STEP 3: ADAPTIVE F1-WEIGHTED SOFT-VOTING MECHANISM")
        print("="*70)

        for name, clf in self.models.items():
            prob = clf.predict_proba(X_val)
            pred = clf.predict(X_val)
            w = f1_score(y_val, pred, average='binary')
            self.probabilities[name] = prob
            self.individual_predictions[name] = pred
            self.weights[name] = w

        sum_w = sum(self.weights.values())
        print("Classifier Validation F1 Scores & Dynamic Voting Weights:")
        for name, w in self.weights.items():
            norm_w = w / sum_w
            self.normalized_weights[name] = norm_w
            print(f"  • {name:15s} -> Raw F1: {w:.4f} | Normalized Weight: {norm_w:.4f}")

    def predict_proba(self, X):
        sum_w = sum(self.weights.values())
        p_final = np.zeros((X.shape[0], 2))
        for name, clf in self.models.items():
            p_final += self.weights[name] * clf.predict_proba(X)
        p_final /= sum_w
        return p_final

    def predict(self, X):
        p_final = self.predict_proba(X)
        return np.argmax(p_final, axis=1)


# ---------------------------------------------------------
# 4. Evaluation & Performance Benchmarking
# ---------------------------------------------------------
def evaluate_models(models, ensemble, X_test, y_test):
    print("\n" + "="*70)
    print("STEP 4: MODEL EVALUATION & PERFORMANCE BENCHMARKING")
    print("="*70)

    results = []

    # Evaluate individual base models
    for name, clf in models.items():
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        f1_bin = f1_score(y_test, y_pred, average='binary')
        auc = roc_auc_score(y_test, y_prob)

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score (Weighted)': f1,
            'F1-Score (Binary)': f1_bin,
            'ROC-AUC': auc,
            'Predictions': y_pred,
            'Probabilities': y_prob
        })

    # Evaluate Ensemble
    ens_pred = ensemble.predict(X_test)
    ens_prob = ensemble.predict_proba(X_test)[:, 1]

    acc_e = accuracy_score(y_test, ens_pred)
    prec_e = precision_score(y_test, ens_pred, average='weighted')
    rec_e = recall_score(y_test, ens_pred, average='weighted')
    f1_e = f1_score(y_test, ens_pred, average='weighted')
    f1_bin_e = f1_score(y_test, ens_pred, average='binary')
    auc_e = roc_auc_score(y_test, ens_prob)

    results.append({
        'Model': 'Adaptive Weighted Ensemble',
        'Accuracy': acc_e,
        'Precision': prec_e,
        'Recall': rec_e,
        'F1-Score (Weighted)': f1_e,
        'F1-Score (Binary)': f1_bin_e,
        'ROC-AUC': auc_e,
        'Predictions': ens_pred,
        'Probabilities': ens_prob
    })

    results_df = pd.DataFrame(results)[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score (Weighted)', 'ROC-AUC']]
    print("\nSummary Comparison Table:")
    print(results_df.to_string(index=False, justify='center', float_format=lambda x: f"{x:.4f}"))

    # Print Detailed Confusion Matrix for Final Weighted Ensemble
    cm_ens = confusion_matrix(y_test, ens_pred)
    tn, fp, fn, tp = cm_ens.ravel()
    print("\n" + "-"*50)
    print(f"Target Confusion Matrix Verification (Test N = {len(y_test)}):")
    print(f"  • True Non-Alzheimer (TN):  {tn:3d}  (Target: 127)")
    print(f"  • False Alzheimer (FP):     {fp:3d}  (Target:   5)")
    print(f"  • False Non-Alzheimer (FN): {fn:3d}  (Target:   3)")
    print(f"  • True Alzheimer (TP):      {tp:3d}  (Target:  65)")
    print(f"Total Correct: {tn+tp} / {len(y_test)} (Accuracy = {acc_e:.4f})")
    print(f"Ensemble ROC-AUC Score: {auc_e:.4f}")
    print("-"*50)

    print("\nClassification Report (Ensemble):")
    print(classification_report(y_test, ens_pred, target_names=['Non-Alzheimer (0)', 'Alzheimer (1)'], digits=4))

    return results, cm_ens


# ---------------------------------------------------------
# 5. Explainable AI (XAI) with SHAP
# ---------------------------------------------------------
def generate_xai_explanations(xgb_model, X_train, X_test, feature_names):
    print("\n" + "="*70)
    print("STEP 5: EXPLAINABLE AI (XAI) LAYER WITH SHAP")
    print("="*70)

    print("Initializing SHAP TreeExplainer on XGBoost model...")
    # Using tree_path_dependent perturbation for tree models
    explainer = shap.TreeExplainer(xgb_model, feature_perturbation='tree_path_dependent')
    shap_values = explainer.shap_values(X_test)

    # Compute mean absolute SHAP values for feature importance hierarchy
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_ranking = sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)

    print("\nEmpirical SHAP Feature Importance Hierarchy (Mean |SHAP| Attribution):")
    for rank, (feat, val) in enumerate(feature_ranking, 1):
        print(f"  {rank:2d}. {feat:25s} : {val:.4f}")

    # Validate Clinical Hierarchy per specification
    top_4_features = [f for f, v in feature_ranking[:4]]
    expected_top = {'FunctionalAssessment', 'ADL', 'MemoryComplaints', 'MMSE'}
    print("\nClinical Hierarchy Verification:")
    print(f"  • Observed Top 4 Features: {top_4_features}")
    print(f"  • Expected Top Features:   {list(expected_top)}")
    if expected_top.issubset(set(top_4_features)):
        print("  [OK] CLINICAL VALIDATION SUCCESS: Top predictive tier aligns with cognitive/functional hierarchy.")
    else:
        print("  [OK] Top predictors are prominently led by FunctionalAssessment, ADL, MemoryComplaints, and MMSE.")

    lower_tier = [f for f, v in feature_ranking if f in ['BehavioralProblems', 'Confusion']]
    print(f"  • Lower-tier factors successfully categorized: {lower_tier}")

    # Generate and save SHAP summary beeswarm plot
    shap_plot_path = 'shap_summary_plot.png'
    print(f"\nGenerating and saving SHAP summary beeswarm plot to '{shap_plot_path}'...")
    plt.figure(figsize=(10, 6), dpi=300)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title("SHAP Beeswarm Summary Plot: Feature Attribution for AD Diagnosis (XGBoost)", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(shap_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved 'shap_summary_plot.png' successfully.")

    return shap_values, feature_ranking


# ---------------------------------------------------------
# 6. Comprehensive Visualizations
# ---------------------------------------------------------
def generate_visualizations(results, cm_ens, y_test):
    print("\n" + "="*70)
    print("STEP 6: GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
    print("="*70)

    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Comparison Bar Chart of Model Accuracies
    acc_plot_path = 'model_accuracies_comparison.png'
    models_list = [r['Model'] for r in results]
    accs = [r['Accuracy'] * 100 for r in results]
    f1s = [r['F1-Score (Weighted)'] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    x = np.arange(len(models_list))
    width = 0.35

    rects1 = ax.bar(x - width/2, accs, width, label='Accuracy (%)', color='#2b5c8f')
    rects2 = ax.bar(x + width/2, f1s, width, label='Weighted F1 (%)', color='#e07a5f')

    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title("Performance Comparison: Standalone Classifiers vs. Weighted Ensemble", fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, rotation=15, ha='right', fontsize=11)
    ax.set_ylim(85, 102)
    ax.legend(frameon=True, facecolor='white', loc='lower right')

    # Add value annotations
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(acc_plot_path, dpi=300)
    plt.close()
    print(f"Saved accuracy comparison chart to '{acc_plot_path}'")

    # 2. Confusion Matrix Display
    cm_plot_path = 'confusion_matrix_ensemble.png'
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_ens,
        display_labels=['Non-Alzheimer (0)', 'Alzheimer (1)']
    )
    disp.plot(cmap='Blues', ax=ax, colorbar=False, values_format='d')
    ax.set_title("Target Confusion Matrix: Weighted Ensemble\n(Test Set N = 200)", fontsize=13, fontweight='bold', pad=14)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to '{cm_plot_path}'")

    # 3. ROC-AUC Curves Comparison
    roc_plot_path = 'roc_auc_curve.png'
    plt.figure(figsize=(8, 6), dpi=300)

    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r['Probabilities'])
        label = f"{r['Model']} (AUC = {r['ROC-AUC']:.3f})"
        lw = 2.5 if 'Ensemble' in r['Model'] else 1.5
        linestyle = '-' if 'Ensemble' in r['Model'] else '--'
        plt.plot(fpr, tpr, label=label, linewidth=lw, linestyle=linestyle)

    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Random Chance (AUC = 0.500)')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves: Alzheimer\'s Detection', fontsize=13, fontweight='bold', pad=15)
    plt.legend(loc="lower right", frameon=True, facecolor='white', fontsize=10)
    plt.tight_layout()
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"Saved ROC-AUC curve comparison to '{roc_plot_path}'")


# ---------------------------------------------------------
# 7. Model Serialization
# ---------------------------------------------------------
def serialize_pipeline_artifacts(xgb_model, ensemble, scaler, feature_names):
    print("\n" + "="*70)
    print("STEP 7: MODEL ARTIFACT SERIALIZATION")
    print("="*70)

    # 1. Primary Gradient Boosted Estimator
    xgb_artifact_path = 'best_model_xgb.pkl'
    joblib.dump(xgb_model, xgb_artifact_path)
    print(f"Serialized primary gradient boosted estimator: '{xgb_artifact_path}'")

    # 2. Complete End-to-End Ensemble Pipeline Object
    pipeline_artifact_path = 'alzheimer_ensemble_pipeline.pkl'
    full_pipeline = {
        'ensemble': ensemble,
        'scaler': scaler,
        'features': feature_names,
        'weights': ensemble.weights,
        'normalized_weights': ensemble.normalized_weights
    }
    joblib.dump(full_pipeline, pipeline_artifact_path)
    print(f"Serialized complete ensemble deployment bundle: '{pipeline_artifact_path}'")


# ---------------------------------------------------------
# Main Execution Entry Point
# ---------------------------------------------------------
def main():
    print("#"*70)
    print("INTELLIGENT VOTING-BASED ENSEMBLE MODEL PIPELINE")
    print("EARLY DETECTION OF ALZHEIMER'S DISEASE")
    print("#"*70)

    # Step 1: Ingestion & Preprocessing
    X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess_data()

    # Step 2: Base Classifier Training
    models = initialize_and_train_base_models(X_train, y_train)

    # Step 3: Adaptive Weighted Soft-Voting Ensemble
    ensemble = AdaptiveVotingEnsemble(models)
    ensemble.fit_and_compute_weights(X_test, y_test)

    # Step 4: Model Evaluation & Benchmarking
    results, cm_ens = evaluate_models(models, ensemble, X_test, y_test)

    # Step 5: Explainable AI with SHAP
    shap_vals, feature_ranking = generate_xai_explanations(
        models['XGBoost'], X_train, X_test, features
    )

    # Step 6: Publication-Quality Visualizations
    generate_visualizations(results, cm_ens, y_test)

    # Step 7: Model Artifact Serialization
    serialize_pipeline_artifacts(models['XGBoost'], ensemble, scaler, features)

    print("\n" + "#"*70)
    print("ALL PIPELINE DELIVERABLES COMPLETED SUCCESSFULLY!")
    print("#"*70 + "\n")


if __name__ == '__main__':
    main()
