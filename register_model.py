import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils import resample
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Dataset yükle
df = pd.read_csv('data/fraud_dataset.csv')
X = df.drop('fraud', axis=1)
y = df['fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("fraud_detection")

# Hyperparameter tuning - 4 farklı kombinasyon dene
param_grid = [
    {"n_estimators": 100, "max_depth": 5,  "min_samples_split": 2},
    {"n_estimators": 200, "max_depth": 10, "min_samples_split": 2},
    {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5},
    {"n_estimators": 300, "max_depth": 10, "min_samples_split": 3},
]

best_f1 = 0
best_run_id = None

print("Hyperparameter tuning başlıyor...\n")

for params in param_grid:
    with mlflow.start_run(run_name=f"tuning_rf_n{params['n_estimators']}_d{params['max_depth']}"):
        mlflow.log_params(params)
        mlflow.set_tag("type", "hyperparameter_tuning")

        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1  = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        mlflow.log_metrics({"f1": f1, "roc_auc": auc})
        mlflow.sklearn.log_model(model, "model")

        print(f"n={params['n_estimators']} depth={params['max_depth']} -> F1:{f1:.3f} AUC:{auc:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_run_id = mlflow.active_run().info.run_id

print(f"\nEn iyi model -> Run ID: {best_run_id}  F1: {best_f1:.3f}")

# Model Registry'e kaydet
model_uri = f"runs:/{best_run_id}/model"
model_name = "fraud_detection_model"

registered = mlflow.register_model(model_uri=model_uri, name=model_name)
print(f"Model kaydedildi: {model_name} v{registered.version}")

# Staging -> Production geçişi
client = MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=registered.version,
    stage="Production"
)
print(f"Model Production'a alındı!")