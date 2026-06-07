import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils import resample
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Load Real Dataset ─────────────────────────────────────────────────────────
df = pd.read_csv('data/creditcard.csv')
df_majority = df[df['Class'] == 0].sample(n=10000, random_state=42)
df_minority = df[df['Class'] == 1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=10000, random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

X = df_balanced.drop('Class', axis=1)
y = df_balanced['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("fraud_detection")

# ── Hyperparameter Search Space (RandomizedSearchCV) ─────────────────────────
param_distributions = {
    'n_estimators':      [50, 100, 200, 300],
    'max_depth':         [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'max_features':      ['sqrt', 'log2'],
}

best_f1 = 0
best_run_id = None

print("Starting hyperparameter tuning (15 trials)...\n")

# ── Run 15 Random Trials ──────────────────────────────────────────────────────
np.random.seed(42)
for trial in range(15):
    params = {
        'n_estimators':      np.random.choice(param_distributions['n_estimators']),
        'max_depth':         np.random.choice(param_distributions['max_depth']),
        'min_samples_split': np.random.choice(param_distributions['min_samples_split']),
        'max_features':      np.random.choice(param_distributions['max_features']),
        'random_state':      42,
    }

    with mlflow.start_run(run_name=f"hyperopt_trial_{trial+1}"):
        mlflow.set_tag("type", "hyperparameter_tuning")
        mlflow.set_tag("trial", str(trial + 1))
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1  = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        mlflow.log_metrics({"f1": f1, "roc_auc": auc})
        mlflow.sklearn.log_model(model, "model")

        print(f"Trial {trial+1:2d} | n={params['n_estimators']} "
              f"depth={params['max_depth']} -> F1: {f1:.3f}  AUC: {auc:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_run_id = mlflow.active_run().info.run_id

print(f"\nBest model -> Run ID: {best_run_id}  F1: {best_f1:.3f}")

# ── Register Best Model ───────────────────────────────────────────────────────
model_name = "fraud_detection_model"
model_uri  = f"runs:/{best_run_id}/model"

registered = mlflow.register_model(model_uri=model_uri, name=model_name)
print(f"Model registered: {model_name} v{registered.version}")

# ── Promote to Production ─────────────────────────────────────────────────────
client = MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=registered.version,
    stage="Production"
)
print("Model promoted to Production!")