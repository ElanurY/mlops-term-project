import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# ── Dataset Generation ──────────────────────────────────────────────────────
np.random.seed(42)
n_samples = 10000

data = {
    'amount':              np.random.exponential(100, n_samples),
    'time_of_day':         np.random.randint(0, 24, n_samples),
    'merchant_category':   np.random.randint(0, 10, n_samples),
    'distance_from_home':  np.random.exponential(50, n_samples),
    'foreign_transaction': np.random.binomial(1, 0.1, n_samples),
    'high_risk_country':   np.random.binomial(1, 0.05, n_samples),
    'fraud':               np.random.binomial(1, 0.05, n_samples),
}

df = pd.DataFrame(data)

# ── Handle Class Imbalance via Oversampling ──────────────────────────────────
df_majority = df[df['fraud'] == 0]
df_minority = df[df['fraud'] == 1]
df_minority_upsampled = resample(
    df_minority, replace=True, n_samples=len(df_majority), random_state=42
)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

df_balanced.to_csv('data/fraud_dataset.csv', index=False)
print("Dataset created (balanced):", df_balanced.shape)
print("Fraud rate:", df_balanced['fraud'].mean())

# ── Train / Test Split ────────────────────────────────────────────────────────
X = df_balanced.drop('fraud', axis=1)
y = df_balanced['fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

mlflow.set_experiment("fraud_detection")

# ── Experiment 1: Logistic Regression ────────────────────────────────────────
with mlflow.start_run(run_name="logistic_regression"):
    params = {"C": 1.0, "max_iter": 200}
    mlflow.log_params(params)

    model = LogisticRegression(**params)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_pred),
    }
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")
    print(f"LR     - F1: {metrics['f1']:.3f}  AUC: {metrics['roc_auc']:.3f}")

# ── Experiment 2: Random Forest (small) ──────────────────────────────────────
with mlflow.start_run(run_name="random_forest_n50"):
    params = {"n_estimators": 50, "max_depth": 5, "random_state": 42}
    mlflow.log_params(params)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_pred),
    }
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")
    print(f"RF50   - F1: {metrics['f1']:.3f}  AUC: {metrics['roc_auc']:.3f}")

# ── Experiment 3: Random Forest (large) ──────────────────────────────────────
with mlflow.start_run(run_name="random_forest_n200"):
    params = {"n_estimators": 200, "max_depth": 10, "random_state": 42}
    mlflow.log_params(params)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_pred),
    }
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")
    print(f"RF200  - F1: {metrics['f1']:.3f}  AUC: {metrics['roc_auc']:.3f}")

print("\nAll experiments completed!")