"""Wrapper de MLflow para as rodadas de simulação do bandit — Etapa 7.1 (Bertelli).

Uma "rodada" é uma combinação de política x hiperparâmetro x seed (ex.: Thompson,
seed=42). Esta função só embrulha uma rodada já calculada — por `train_bandit` (Etapa 3,
Adryen) ou por uma simulação provisória própria — num run do MLflow, registrando os
parâmetros e métricas que o enunciado exige (priors/epsilon/seed documentados).

Usa sempre `bm.tracking.setup_mlflow`, nunca `mlflow.set_tracking_uri` direto — é ela que
decide entre o servidor compartilhado do grupo (se houver `.env`) e o SQLite local.
"""

from typing import Any

import mlflow

from bm.tracking import setup_mlflow

EXPERIMENT_NAME = "datathon-bandit"


def log_bandit_run(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: list[str] | None = None,
    experiment: str = EXPERIMENT_NAME,
) -> str:
    """Registra uma rodada no MLflow e devolve o `run_id` (útil pra citar no README/vídeo).

    `params` deve sempre incluir os priors/epsilon/seed da rodada — é uma exigência
    textual do enunciado, não só boa prática.
    """
    setup_mlflow(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for artifact_path in artifacts or []:
            mlflow.log_artifact(artifact_path)
        return run.info.run_id
