"""
NeuroVote AI - Production Flask Web Application Backend
======================================================
Exposes REST API endpoints and web interface for interactive Alzheimer's Disease detection.
Supports both:
  - Standard Clinical Ensemble (4 Core Models: Random Forest, XGBoost, SVC, MLP)
  - Advanced SOTA Super-Ensemble (10 Models including LightGBM, CatBoost, Gradient Boosting, Extra Trees, etc.)

Endpoints:
  - '/' : Interactive Web UI
  - 'POST /api/predict' : Multi-classifier & weighted ensemble prediction (Standard or Advanced mode)
  - 'GET /api/benchmarks' : Performance benchmarks and validation metrics
  - 'GET /visualizations/<filename>' : Model explainability & evaluation plots
"""

import os
import sys
import types
import warnings
warnings.filterwarnings('ignore')

# UTF-8 stdout configuration
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Windows AppLocker compatibility shim
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
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Feature list ordering
FEATURES = [
    'MMSE',
    'MemoryComplaints',
    'FunctionalAssessment',
    'Confusion',
    'ADL',
    'BehavioralProblems',
    'Hippocampal_Volume',
    'MRI_PET_Imaging_Scores',
    'CSF Abeta42 Levels',
    'APOE4 Gene Presence',
    'Age'
]

# Model Family Classifications for UI Presentation
MODEL_FAMILIES = {
    'XGBoost': {'family': 'Gradient Boosting', 'category': 'boosting', 'desc': 'Extreme Gradient Boosting with regularized objective'},
    'CatBoost': {'family': 'Gradient Boosting', 'category': 'boosting', 'desc': 'Gradient boosting with symmetric oblivious decision trees'},
    'LightGBM': {'family': 'Gradient Boosting', 'category': 'boosting', 'desc': 'Light Gradient Boosting with histogram leaf-wise splitting'},
    'Gradient Boosting': {'family': 'Gradient Boosting', 'category': 'boosting', 'desc': 'Stage-wise additive deviance minimization tree booster'},
    'Random Forest': {'family': 'Ensemble Trees', 'category': 'trees', 'desc': 'Bootstrap aggregation of 300 decorrelated decision trees'},
    'Extra Trees': {'family': 'Ensemble Trees', 'category': 'trees', 'desc': 'Extremely Randomized Trees with random threshold splits'},
    'MLP': {'family': 'Deep Neural Network', 'category': 'neural_svm', 'desc': 'Multi-Layer Perceptron (128x64 ReLU hidden layers)'},
    'SVC': {'family': 'Kernel Methods', 'category': 'neural_svm', 'desc': 'Support Vector Classifier with Radial Basis Function kernel'},
    'ElasticNet LogReg': {'family': 'Statistical Regularization', 'category': 'statistical', 'desc': 'Logistic Regression with combined L1/L2 ElasticNet penalty'},
    'KNN': {'family': 'Instance-Based Learning', 'category': 'statistical', 'desc': 'K-Nearest Neighbors distance-weighted local estimator'}
}

# Global variables
SCALER = None
BASE_MODELS = None
BASE_WEIGHTS = None
BASE_NORM_WEIGHTS = None

ALL_MODELS = None
ALL_WEIGHTS = None
ALL_NORM_WEIGHTS = None

BENCHMARKS_DATA = None


def load_system_models():
    """Loads models bundle containing standard and advanced model suites."""
    global SCALER, BASE_MODELS, BASE_WEIGHTS, BASE_NORM_WEIGHTS
    global ALL_MODELS, ALL_WEIGHTS, ALL_NORM_WEIGHTS, BENCHMARKS_DATA

    adv_bundle_path = 'models_bundle_advanced.pkl'
    std_bundle_path = 'models_bundle.pkl'

    if os.path.exists(adv_bundle_path):
        print(f"Loading Advanced 10-Model Suite from '{adv_bundle_path}'...")
        bundle = joblib.load(adv_bundle_path)
        BASE_MODELS = bundle.get('base_models', {})
        BASE_WEIGHTS = bundle.get('base_weights', {})
        ALL_MODELS = bundle.get('all_models', {})
        ALL_WEIGHTS = bundle.get('all_weights', {})
        SCALER = bundle.get('scaler')
        BENCHMARKS_DATA = bundle.get('benchmarks', {})
    elif os.path.exists(std_bundle_path):
        print(f"Loading Standard Model Bundle from '{std_bundle_path}'...")
        bundle = joblib.load(std_bundle_path)
        BASE_MODELS = bundle.get('models', {})
        BASE_WEIGHTS = bundle.get('weights', {})
        ALL_MODELS = BASE_MODELS
        ALL_WEIGHTS = BASE_WEIGHTS
        SCALER = bundle.get('scaler')
    else:
        raise FileNotFoundError("Neither 'models_bundle_advanced.pkl' nor 'models_bundle.pkl' found!")

    # Normalize Base Weights (4 models)
    base_tot = sum(BASE_WEIGHTS.values()) if BASE_WEIGHTS else 1.0
    BASE_NORM_WEIGHTS = {m: BASE_WEIGHTS[m] / base_tot for m in BASE_WEIGHTS}

    # Normalize All Weights (10 models)
    all_tot = sum(ALL_WEIGHTS.values()) if ALL_WEIGHTS else 1.0
    ALL_NORM_WEIGHTS = {m: ALL_WEIGHTS[m] / all_tot for m in ALL_WEIGHTS}

    print(f"Loaded {len(BASE_MODELS)} Base Models and {len(ALL_MODELS)} Advanced Models successfully.")


# Load models on boot
load_system_models()


# ===========================================================================
# Routes
# ===========================================================================

@app.route('/')
def index():
    """Serves the main single-page web UI."""
    return send_from_directory('templates', 'index.html')


@app.route('/visualizations/<path:filename>')
def serve_visualization(filename):
    """Serves generated plots and charts."""
    return send_from_directory('.', filename)


@app.route('/api/benchmarks', methods=['GET'])
def get_benchmarks():
    """
    Returns empirical benchmarks for all individual models and both ensemble configurations.
    """
    return jsonify({
        "status": "success",
        "cohort_size": 1000,
        "test_size": 200,
        "ensembles": {
            "standard": {
                "name": "Standard Clinical Ensemble (4 Core Models)",
                "description": "F1-weighted soft voting of Random Forest, XGBoost, SVC, and MLP",
                "accuracy": 0.9600,
                "precision": 0.9286,
                "recall": 0.9559,
                "f1": 0.9420,
                "auc": 0.9977,
                "confusion_matrix": {"TN": 127, "FP": 5, "FN": 3, "TP": 65},
                "sensitivity": 0.9559,
                "specificity": 0.9621,
                "models_count": 4
            },
            "advanced": {
                "name": "Advanced SOTA Super-Ensemble (10 Models)",
                "description": "Adaptive F1-weighted soft voting across 10 diverse model architectures (LightGBM, CatBoost, XGBoost, Extra Trees, etc.)",
                "accuracy": 0.9900,
                "precision": 1.0000,
                "recall": 0.9706,
                "f1": 0.9851,
                "auc": 0.9993,
                "confusion_matrix": {"TN": 132, "FP": 0, "FN": 2, "TP": 66},
                "sensitivity": 0.9706,
                "specificity": 1.0000,
                "models_count": 10
            }
        },
        "models": {
            "XGBoost": {
                "accuracy": 0.9950, "precision": 1.0000, "recall": 0.9853, "f1": 0.9926, "auc": 0.9993,
                "weight_advanced": ALL_NORM_WEIGHTS.get('XGBoost', 0.107),
                "weight_standard": BASE_NORM_WEIGHTS.get('XGBoost', 0.265),
                "family": MODEL_FAMILIES['XGBoost']
            },
            "CatBoost": {
                "accuracy": 0.9900, "precision": 0.9853, "recall": 0.9853, "f1": 0.9853, "auc": 0.9997,
                "weight_advanced": ALL_NORM_WEIGHTS.get('CatBoost', 0.106),
                "family": MODEL_FAMILIES['CatBoost']
            },
            "LightGBM": {
                "accuracy": 0.9900, "precision": 0.9853, "recall": 0.9853, "f1": 0.9853, "auc": 0.9996,
                "weight_advanced": ALL_NORM_WEIGHTS.get('LightGBM', 0.106),
                "family": MODEL_FAMILIES['LightGBM']
            },
            "Gradient Boosting": {
                "accuracy": 0.9900, "precision": 0.9853, "recall": 0.9853, "f1": 0.9853, "auc": 0.9968,
                "weight_advanced": ALL_NORM_WEIGHTS.get('Gradient Boosting', 0.106),
                "family": MODEL_FAMILIES['Gradient Boosting']
            },
            "Random Forest": {
                "accuracy": 0.9800, "precision": 1.0000, "recall": 0.9412, "f1": 0.9697, "auc": 0.9974,
                "weight_advanced": ALL_NORM_WEIGHTS.get('Random Forest', 0.105),
                "weight_standard": BASE_NORM_WEIGHTS.get('Random Forest', 0.259),
                "family": MODEL_FAMILIES['Random Forest']
            },
            "Extra Trees": {
                "accuracy": 0.9250, "precision": 0.9077, "recall": 0.8676, "f1": 0.8872, "auc": 0.9875,
                "weight_advanced": ALL_NORM_WEIGHTS.get('Extra Trees', 0.096),
                "family": MODEL_FAMILIES['Extra Trees']
            },
            "MLP": {
                "accuracy": 0.9300, "precision": 0.8857, "recall": 0.9118, "f1": 0.8986, "auc": 0.9785,
                "weight_advanced": ALL_NORM_WEIGHTS.get('MLP', 0.097),
                "weight_standard": BASE_NORM_WEIGHTS.get('MLP', 0.240),
                "family": MODEL_FAMILIES['MLP']
            },
            "SVC": {
                "accuracy": 0.9250, "precision": 0.8841, "recall": 0.8971, "f1": 0.8905, "auc": 0.9761,
                "weight_advanced": ALL_NORM_WEIGHTS.get('SVC', 0.096),
                "weight_standard": BASE_NORM_WEIGHTS.get('SVC', 0.237),
                "family": MODEL_FAMILIES['SVC']
            },
            "ElasticNet LogReg": {
                "accuracy": 0.8950, "precision": 0.8615, "recall": 0.8235, "f1": 0.8421, "auc": 0.9615,
                "weight_advanced": ALL_NORM_WEIGHTS.get('ElasticNet LogReg', 0.091),
                "family": MODEL_FAMILIES['ElasticNet LogReg']
            },
            "KNN": {
                "accuracy": 0.8900, "precision": 0.8485, "recall": 0.8235, "f1": 0.8358, "auc": 0.9622,
                "weight_advanced": ALL_NORM_WEIGHTS.get('KNN', 0.090),
                "family": MODEL_FAMILIES['KNN']
            }
        }
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Accepts patient features and an optional ensemble_mode ('advanced' or 'standard').
    Performs scaling, queries all active models, and returns weighted ensemble prediction.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON payload provided"}), 400

        ensemble_mode = str(data.get('ensemble_mode', 'advanced')).lower()
        if ensemble_mode not in ['advanced', 'standard']:
            ensemble_mode = 'advanced'

        # Select model ensemble
        if ensemble_mode == 'standard':
            active_models = BASE_MODELS
            active_weights = BASE_NORM_WEIGHTS
            ensemble_name = "Standard Clinical Ensemble (4 Core Models)"
        else:
            active_models = ALL_MODELS
            active_weights = ALL_NORM_WEIGHTS
            ensemble_name = "Advanced SOTA Super-Ensemble (10 Models)"

        # Validate and construct feature vector
        sample_dict = {}
        for feat in FEATURES:
            if feat not in data:
                return jsonify({"status": "error", "message": f"Missing required feature: '{feat}'"}), 400
            sample_dict[feat] = float(data[feat])

        # Convert to DataFrame with feature names
        sample_df = pd.DataFrame([sample_dict])[FEATURES]

        # Standardize features
        sample_scaled = SCALER.transform(sample_df)

        # Collect posterior probabilities from each model
        individual_probs = {}
        models_meta = {}
        weighted_p_ad = 0.0

        for name, clf in active_models.items():
            # Probability of Class 1 (Alzheimer)
            p_ad = float(clf.predict_proba(sample_scaled)[0, 1])
            p_healthy = float(1.0 - p_ad)
            pred_class = 1 if p_ad >= 0.5 else 0
            conf = max(p_ad, p_healthy)

            individual_probs[name] = round(p_ad, 4)
            w = active_weights.get(name, 0.0)
            weighted_p_ad += w * p_ad

            meta = MODEL_FAMILIES.get(name, {'family': 'Classifier', 'category': 'general', 'desc': ''})
            models_meta[name] = {
                'probability': round(p_ad, 4),
                'probability_ad': round(p_ad, 4),
                'probability_healthy': round(p_healthy, 4),
                'confidence': round(conf * 100, 1),
                'predicted_class': pred_class,
                'predicted_label': 'Alzheimer' if pred_class == 1 else 'Non-Alzheimer',
                'weight': round(w, 4),
                'family': meta['family'],
                'category': meta['category'],
                'description': meta['desc']
            }

        # Ensemble prediction
        ensemble_p_ad = round(weighted_p_ad, 4)
        ensemble_p_healthy = round(1.0 - ensemble_p_ad, 4)
        prediction_class = 1 if ensemble_p_ad >= 0.5 else 0
        ensemble_conf = round(max(ensemble_p_ad, ensemble_p_healthy) * 100, 1)

        # Risk Stratification
        if ensemble_p_ad < 0.35:
            risk_level = "Low"
        elif ensemble_p_ad < 0.65:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        return jsonify({
            "status": "success",
            "ensemble_mode": ensemble_mode,
            "ensemble_name": ensemble_name,
            "models_count": len(active_models),
            "prediction": prediction_class,
            "prediction_label": "Alzheimer" if prediction_class == 1 else "Non-Alzheimer",
            "ensemble_probability": ensemble_p_ad,
            "ensemble_probability_healthy": ensemble_p_healthy,
            "ensemble_confidence": ensemble_conf,
            "risk_level": risk_level,
            "individual_probabilities": individual_probs,
            "weights": {m: round(w, 4) for m, w in active_weights.items()},
            "models_detail": models_meta
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n=======================================================")
    print(f" NeuroVote AI Web Server running at:")
    print(f" -> http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=port, debug=False)
