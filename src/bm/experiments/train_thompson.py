"""Treina o Thompson Sampling contextual e salva `models/thompson.joblib`.

Este é o artefato que a API (Etapa 5) e o golden set (Etapa 4.2) carregam.

O treino usa **apenas o split de treino** (primeiros 80%, cronológico). Os últimos 20%
ficam de fora de propósito: é de lá que `bm.golden_set` tira os 5 clientes, e o golden
set só vale como teste de regressão se o bandit nunca tiver visto aquelas linhas.

Rodar: `uv run python -m bm.experiments.train_thompson`
"""

from pathlib import Path

import joblib
import mlflow
import pandas as pd

from bm.data_prep import temporal_split
from bm.experiments.train import train_bandit
from bm.models.bandit import ThompsonSampling
from bm.tracking import setup_mlflow

ALPHA = 1.0
N_FEATURES = 59
SEED = 7

setup_mlflow("datathon-train-thompson")

print("Tracking URI:", mlflow.get_tracking_uri())
print("Experiment:", mlflow.get_experiment_by_name("datathon-train-thompson"))

frame = pd.read_parquet(Path("data/processed/bandit_frame.parquet"))
treino, _teste = temporal_split(frame)

preprocessor = joblib.load(Path("models/preprocessor.joblib"))

bandit = ThompsonSampling(alpha=ALPHA, n_features=N_FEATURES, seed=SEED)

with mlflow.start_run(run_name="thompson-treino-80pct"):
    # log_to_mlflow=False: o log por linha seriam ~2 chamadas x dezenas de milhares de
    # rodadas contra o servidor compartilhado do time. As métricas que importam são as
    # agregadas, logadas abaixo.
    _, rewards_matched = train_bandit(bandit, treino, preprocessor, log_to_mlflow=False)

    mlflow.log_params(
        {
            "algorithm": "Linear Thompson Sampling (contextual)",
            "alpha": ALPHA,
            "prior": "Normal(0, alpha^2 * I) — nao-informativo, nenhum braco favorecido",
            "n_features": N_FEATURES,
            "seed": SEED,
            "split": "temporal 80/20, sem shuffle (treino = primeiros 80%)",
            "n_linhas_treino": len(treino),
        }
    )

    conversao = sum(rewards_matched) / len(rewards_matched) if rewards_matched else 0.0
    mlflow.log_metrics(
        {"conversao_replay": conversao, "n_matched": len(rewards_matched)}
    )

    model_path = Path("models/thompson.joblib")
    joblib.dump(bandit, model_path)
    mlflow.log_artifact(str(model_path), artifact_path="models")

print(
    f"treinado em {len(treino)} linhas (80% cronologico); "
    f"{len(rewards_matched)} casaram no replay; conversao {conversao:.2%}"
)
print(f"salvo em {model_path}")
