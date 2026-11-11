import os
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
  import shap

  SHAP_AVAILABLE = True
except Exception:
  SHAP_AVAILABLE = False
  print('Note: shap unavailable. Using Permutation Importance instead.')

# =====================================================================
# 1. LOAD DATASET (Automatically resolves your script's folder)
# =====================================================================
import kagglehub

URL = kagglehub.dataset_download("hasibur013/diabetes-dataset")
data = pd.read_csv(URL)

print(data.head())

# =====================================================================
# 2. DATA PREPROCESSING
# =====================================================================

# Remove duplicate rows
print('Number of duplicate rows found: ', data.duplicated().sum())
data = data.drop_duplicates()
print('Shape after removing duplicates: ', data.shape)

# Clean biologically impossible zeros
zero_invalid_cols = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
]
for col in zero_invalid_cols:
  if col in data.columns:
    data[col] = data[col].replace(0, np.nan)

print('Numbers of empty cell : ', data.isnull().any().sum())
print(data.isna().sum())

# Remove outliers (IQR method)
outlier_columns = [col for col in data.columns if col != 'Outcome']


def remove_outliers_iqr(df, columns):
  df_clean = df.copy()
  for col in columns:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean = df_clean[
        (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
    ]
  return df_clean


print('Shape before outlier removal: ', data.shape)
data = remove_outliers_iqr(data, outlier_columns)
print('Shape after outlier removal: ', data.shape)

# =====================================================================
# 3. LEAK-FREE SPLITTING, IMPUTATION, BALANCING & SCALING
# =====================================================================
algorithms = [
    'RandomForest',
    'LogisticRegression',
    'SGDClassifier',
    'NeuralNetwork',
    'Calibrated_HGB_Default',
    'Calibrated_HGB_Clinical',
]
accuracies = []
precisions = []
recalls = []
f1_scores = []
roc_auc_scores = []


def append_metrics(accuracy, precision, recall, f1, roc_auc):
  accuracies.append(accuracy)
  precisions.append(precision)
  recalls.append(recall)
  f1_scores.append(f1)
  roc_auc_scores.append(roc_auc)


y = data['Outcome']
x = data.drop('Outcome', axis=1)

# STEP A: Split FIRST before imputing, oversampling, or scaling
X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=25
)

# STEP B: Impute missing values (fit ONLY on training data)
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
X_train_imp = knn_imputer.fit_transform(X_train_raw)
X_test_imp = knn_imputer.transform(X_test_raw)

# STEP C: Oversample ONLY the training data (X_test remains 100% untouched)
print('Training class distribution before balancing:\n', y_train_raw.value_counts())
ros = RandomOverSampler(random_state=25)
X_train_bal, y_train = ros.fit_resample(X_train_imp, y_train_raw)
print(
    'Training class distribution after balancing:\n',
    pd.Series(y_train).value_counts(),
)

# STEP D: Scale features (fit ONLY on training data)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_bal)
X_test = scaler.transform(X_test_imp)

print('Final X_train shape:', X_train.shape, '| Unseen X_test shape:', X_test.shape)

# =====================================================================
# 4. BUILDING & TRAINING ALL MODELS
# =====================================================================
model_1 = RandomForestClassifier(random_state=25)
model_2 = LogisticRegression(random_state=25)
# loss='log_loss' enables predict_proba() on SGDClassifier for true probability ROC-AUC
model_3 = SGDClassifier(loss='log_loss', random_state=25)
model_4 = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=25,
)

# Calibrated HistGradientBoosting Pipeline
preprocessor = Pipeline([
    ('scaler', RobustScaler()),
])

base_model = HistGradientBoostingClassifier(
    max_iter=150,
    max_depth=4,
    learning_rate=0.05,
    l2_regularization=1.0,
    random_state=42,
)

calibrated_clf = CalibratedClassifierCV(
    estimator=base_model,
    method='sigmoid',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
)

full_pipeline = Pipeline(
    [('prep', preprocessor), ('classifier', calibrated_clf)]
)

# Train all models
model_1.fit(X_train, y_train)
model_2.fit(X_train, y_train)
model_3.fit(X_train, y_train)
model_4.fit(X_train, y_train)
full_pipeline.fit(X_train, y_train)

# =====================================================================
# 5. PREDICTING & EVALUATING ALL MODELS 
# =====================================================================

# 1. Random Forest
rf_probs = model_1.predict_proba(X_test)[:, 1]
target_randomForest = model_1.predict(X_test)
random_accuracy = accuracy_score(y_test, target_randomForest)
random_precision, random_recall, random_f1, _ = precision_recall_fscore_support(
    y_test, target_randomForest, average='binary'
)
random_auc = roc_auc_score(y_test, rf_probs)
append_metrics(
    random_accuracy, random_precision, random_recall, random_f1, random_auc
)

# 2. Logistic Regression
lr_probs = model_2.predict_proba(X_test)[:, 1]
target_LR = model_2.predict(X_test)
LR_accuracy = accuracy_score(y_test, target_LR)
LR_precision, LR_recall, LR_f1, _ = precision_recall_fscore_support(
    y_test, target_LR, average='binary'
)
LR_auc = roc_auc_score(y_test, lr_probs)
append_metrics(LR_accuracy, LR_precision, LR_recall, LR_f1, LR_auc)

# 3. SGDClassifier
sgdc_probs = model_3.predict_proba(X_test)[:, 1]
target_SGDC = model_3.predict(X_test)
SGDC_accuracy = accuracy_score(y_test, target_SGDC)
SGDC_precision, SGDC_recall, SGDC_f1, _ = precision_recall_fscore_support(
    y_test, target_SGDC, average='binary'
)
SGDC_auc = roc_auc_score(y_test, sgdc_probs)
append_metrics(SGDC_accuracy, SGDC_precision, SGDC_recall, SGDC_f1, SGDC_auc)

# 4. Neural Network
nn_probs = model_4.predict_proba(X_test)[:, 1]
target_NN = model_4.predict(X_test)
NN_accuracy = accuracy_score(y_test, target_NN)
NN_precision, NN_recall, NN_f1, _ = precision_recall_fscore_support(
    y_test, target_NN, average='binary'
)
NN_auc = roc_auc_score(y_test, nn_probs)
append_metrics(NN_accuracy, NN_precision, NN_recall, NN_f1, NN_auc)

# 5. Calibrated HistGradientBoosting (Default 0.50 vs. Youden's J Clinical Threshold)
train_probs = full_pipeline.predict_proba(X_train)[:, 1]
fpr, tpr, thresholds = roc_curve(y_train, train_probs)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

test_probs = full_pipeline.predict_proba(X_test)[:, 1]
test_preds_default = (test_probs >= 0.50).astype(int)
test_preds_optimal = (test_probs >= optimal_threshold).astype(int)

HGB_auc = roc_auc_score(y_test, test_probs)
HGB_brier = brier_score_loss(y_test, test_probs)

HGB_def_acc = accuracy_score(y_test, test_preds_default)
HGB_def_prec, HGB_def_rec, HGB_def_f1, _ = precision_recall_fscore_support(
    y_test, test_preds_default, average='binary'
)
append_metrics(HGB_def_acc, HGB_def_prec, HGB_def_rec, HGB_def_f1, HGB_auc)

HGB_opt_acc = accuracy_score(y_test, test_preds_optimal)
HGB_opt_prec, HGB_opt_rec, HGB_opt_f1, _ = precision_recall_fscore_support(
    y_test, test_preds_optimal, average='binary'
)
append_metrics(HGB_opt_acc, HGB_opt_prec, HGB_opt_rec, HGB_opt_f1, HGB_auc)

print('=========================================================')
print('    CALIBRATED HISTGRADIENTBOOSTING DIAGNOSTICS')
print('=========================================================')
print(f'HGB ROC-AUC Score               :  {HGB_auc:.4f}')
print(f'HGB Brier Score (Calibration)   :  {HGB_brier:.4f}')
print(f'Optimal Clinical Threshold      :  {optimal_threshold:.4f}\n')
print(f'--- Clinical Threshold ({optimal_threshold:.2f}) Report ---')
print(classification_report(y_test, test_preds_optimal))
print('Confusion Matrix (Clinical Threshold):')
print(confusion_matrix(y_test, test_preds_optimal))

# Overall Comparison Table
comparison_df = pd.DataFrame({
    'Algorithm': algorithms,
    'Accuracy': accuracies,
    'Precision': precisions,
    'Recall': recalls,
    'F1_Score': f1_scores,
    'ROC_AUC': roc_auc_scores,
})
print('\n=========================================================')
print('          OVERALL LEAK-FREE MODEL COMPARISON')
print('=========================================================')
print(comparison_df.sort_values(by='Accuracy', ascending=False))
# =====================================================================
# PASTE THE 10-FOLD STRATIFIED CROSS-VALIDATION CODE HERE
# =====================================================================
models_cv = {
    'LogisticRegression': LogisticRegression(random_state=25),
    'RandomForest': RandomForestClassifier(random_state=25),
    'Calibrated_HGB': CalibratedClassifierCV(
        estimator=HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=4,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=42,
        ),
        method='sigmoid',
        cv=3,
    ),
    'NeuralNetwork': MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=25,
    ),
    'SGDClassifier': SGDClassifier(loss='log_loss', random_state=25),
}

# 2. Set up 10-Fold Stratified Cross-Validation
cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=25)
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

cv_summary = []

# 3. Run leak-free 10-fold CV for each model using x and y (before train_test_split)
for name, clf in models_cv.items():
    pipeline = ImbPipeline([
        ('imputer', KNNImputer(n_neighbors=5, weights='distance')),
        ('oversampler', RandomOverSampler(random_state=25)),
        ('scaler', StandardScaler()),
        ('model', clf),
    ])

    scores = cross_validate(
        pipeline, x, y, cv=cv_strategy, scoring=scoring_metrics, n_jobs=-1
    )

    cv_summary.append({
        'Algorithm': name,
        'Accuracy': f"{scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}",
        'Precision': f"{scores['test_precision'].mean():.4f} ± {scores['test_precision'].std():.4f}",
        'Recall': f"{scores['test_recall'].mean():.4f} ± {scores['test_recall'].std():.4f}",
        'F1_Score': f"{scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}",
        'ROC_AUC': f"{scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}",
    })

cv_results_df = pd.DataFrame(cv_summary)

print('\n=============================================================================')
print('              10-FOLD STRATIFIED CROSS-VALIDATION (MEAN ± STD)')
print('=============================================================================')
print(cv_results_df.sort_values(by='Accuracy', ascending=False))

# =====================================================================
# 6. EXPLAINABLE AI (SHAP OR PERMUTATION FALLBACK)
# =====================================================================
X_train_processed = pd.DataFrame(
    full_pipeline.named_steps['prep'].transform(X_train), columns=x.columns
)
X_test_processed = pd.DataFrame(
    full_pipeline.named_steps['prep'].transform(X_test), columns=x.columns
)
X_test_original_units = pd.DataFrame(
    scaler.inverse_transform(X_test), columns=x.columns
)
if SHAP_AVAILABLE:
    print('\nGenerating SHAP explanations for clinical interpretability...')
    explain_model = base_model.fit(X_train_processed, y_train)
    explainer = shap.TreeExplainer(explain_model)
    shap_values = explainer(X_test_processed)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_processed, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300)
    print('Saved global SHAP feature importance plot to: shap_summary.png')
    patient_idx = 0
    print(f'\n--- Patient #{patient_idx} Risk Breakdown ---')
    print(f'Predicted Diabetes Risk: {test_probs[patient_idx]:.1%}')
    for feature, val, impact in zip(
        x.columns,
        X_test_original_units.iloc[patient_idx],
        shap_values.values[patient_idx],
    ):
        direction = '+' if impact > 0 else ''
        print(f'  {feature:<26} | Raw Value: {val:<8.2f} | SHAP Impact: {direction}{impact:.4f}')
else:
    print('\nGenerating Permutation Feature Importance...')
    perm = permutation_importance(
        full_pipeline, X_test, y_test, n_repeats=15, random_state=42, scoring='roc_auc'
    )
    for feature, imp in sorted(zip(x.columns, perm.importances_mean), key=lambda item: item[1], reverse=True):
        print(f'  {feature:<26} | Mean AUC Drop: {imp:.4f}')