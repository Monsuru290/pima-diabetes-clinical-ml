# Clinical Diabetes Risk Prediction & Explainable AI (Pima Dataset)

A leak-free, clinically calibrated machine learning benchmark on the 768-record **Pima Indians Diabetes Dataset**. This project compares standard classifiers against a **Calibrated Histogram-Based Gradient Boosting (`HistGradientBoostingClassifier`)** model with **Youden's J clinical threshold optimization** and **SHAP / Permutation feature attribution**.

## Key Features
- **Strict Leakage Prevention:** Train/test splitting is performed *before* `KNNImputer` missing-value estimation, `RandomOverSampler` class balancing, and `StandardScaler` normalization.
- **Probability Calibration:** Uses sigmoid calibration (`CalibratedClassifierCV`) evaluated via Brier Score so predicted risk percentages reflect true clinical likelihood.
- **Dual Validation Protocol:** Evaluates models on both an **80/20 Stratified Holdout Split** and **10-Fold Stratified Cross-Validation (`ImbPipeline`)**.
- **Explainable AI (XAI):** Generates global feature attribution plots (`shap_summary.png`) and single-patient risk breakdowns.

## Benchmark Results

### 1. 80/20 Stratified Holdout Split

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LogisticRegression** | **0.8636** | 0.7273 | **0.8421** | **0.7805** | **0.9104** |
| **RandomForest** | 0.8485 | **0.8000** | 0.6316 | 0.7059 | 0.9054 |
| **Calibrated_HGB_Default** | 0.8333 | 0.7222 | 0.6842 | 0.7027 | 0.8555 |
| **Calibrated_HGB_Clinical** | 0.8182 | 0.7059 | 0.6316 | 0.6667 | 0.8555 |
| **NeuralNetwork** | 0.8030 | 0.6875 | 0.5789 | 0.6286 | 0.8186 |
| **SGDClassifier** | 0.7576 | 0.5600 | 0.7368 | 0.6364 | 0.8914 |

### 2. 10-Fold Stratified Cross-Validation (Mean ± Std)

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RandomForest** | **0.7970 ± 0.0899** | **0.6783 ± 0.1965** | 0.6144 ± 0.1363 | 0.6365 ± 0.1492 | **0.8409 ± 0.1367** |
| **Calibrated_HGB** | 0.7848 ± 0.0784 | 0.6319 ± 0.1451 | 0.6156 ± 0.1766 | 0.6146 ± 0.1418 | 0.8323 ± 0.1371 |
| **LogisticRegression** | 0.7788 ± 0.0968 | 0.6035 ± 0.1642 | 0.7100 ± 0.1796 | 0.6457 ± 0.1580 | 0.8314 ± 0.1040 |
| **NeuralNetwork** | 0.7758 ± 0.0815 | 0.5994 ± 0.1377 | **0.7122 ± 0.1316** | **0.6460 ± 0.1242** | 0.8056 ± 0.1274 |
| **SGDClassifier** | 0.7273 ± 0.1342 | 0.5592 ± 0.2061 | 0.6889 ± 0.1373 | 0.6049 ± 0.1603 | 0.8132 ± 0.1123 |

## How to Run
```bash
git clone https://github.com<your-username>/pima-diabetes-clinical-ml.git
cd pima-diabetes-clinical-ml
pip install -r requirements.txt
python diabetes_pipeline.py
```
