from src.bm.tracking import setup_mlflow
from src.bm.experiments.train import train_bandit
from src.bm.models.bandit import ThompsonSampling
from src.bm.data_prep import CONTEXT_FEATURES
import pandas as pd
import joblib
from pathlib import Path
import mlflow

setup_mlflow("datathon-train-thompson")

print("Tracking URI:", mlflow.get_tracking_uri())
print("Experiment:", mlflow.get_experiment_by_name("datathon-train-thompson"))

frame = pd.read_parquet(Path("data/processed/bandit_frame.parquet"))

preprocessor = joblib.load(Path("models/preprocessor.joblib"))

bandit = ThompsonSampling(alpha=1.0, n_features=59, seed=7)

with mlflow.start_run(run_name="baseline"):
    train_bandit(
        bandit,
        frame,
        preprocessor
    )

    mlflow.log_param("algorithm", "Linear Thompson Sampling")
    mlflow.log_param("alpha", bandit.alpha)
    mlflow.log_param("n_features", bandit.n_features)

    model_path = Path("models/thompson.joblib")
    joblib.dump(bandit, model_path)

    mlflow.log_artifact("models/thompson.joblib", artifact_path="models")