import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Production modelini yükle
model_name = "fraud_detection_model"
model_uri = f"models:/{model_name}/Production"
model = mlflow.sklearn.load_model(model_uri)
print(f"Model yüklendi: {model_name} @ Production\n")

mlflow.set_experiment("fraud_detection_monitoring")

# Simüle edilmiş 3 farklı zaman dilimi (drift simülasyonu)
np.random.seed(99)

batches = {
    "week_1_normal":  {"noise": 0.0, "fraud_rate": 0.5},
    "week_2_slight":  {"noise": 0.3, "fraud_rate": 0.5},
    "week_3_drift":   {"noise": 0.8, "fraud_rate": 0.5},
}

for batch_name, cfg in batches.items():
    n = 2000
    X_new = pd.DataFrame({
        'amount':             np.random.exponential(100, n) + cfg["noise"] * np.random.randn(n) * 50,
        'time_of_day':        np.random.randint(0, 24, n),
        'merchant_category':  np.random.randint(0, 10, n),
        'distance_from_home': np.random.exponential(50, n) + cfg["noise"] * np.random.randn(n) * 20,
        'foreign_transaction':np.random.binomial(1, 0.1, n),
        'high_risk_country':  np.random.binomial(1, 0.05, n),
    })
    y_new = np.random.binomial(1, cfg["fraud_rate"], n)

    y_pred = model.predict(X_new)

    f1  = f1_score(y_new, y_pred)
    auc = roc_auc_score(y_new, y_pred)
    acc = accuracy_score(y_new, y_pred)

    with mlflow.start_run(run_name=f"monitoring_{batch_name}"):
        mlflow.set_tag("monitoring", "true")
        mlflow.set_tag("batch", batch_name)
        mlflow.log_metrics({"f1": f1, "roc_auc": auc, "accuracy": acc})

    print(f"{batch_name:25s} -> F1:{f1:.3f}  AUC:{auc:.3f}  ACC:{acc:.3f}")

print("\nMonitoring tamamlandı!")