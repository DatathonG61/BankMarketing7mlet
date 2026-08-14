"""Roda o Thompson Sampling em múltiplas seeds -- Etapa 4.1 e 7.1 (Bertelli).

Não implementa nenhum algoritmo novo: só chama o que já existe em `bm.models.bandit`
e `bm.experiments.train` (Etapa 3, Adryen) várias vezes com seeds diferentes, porque
bandit é estocástico e uma rodada única não é resultado (exigência do plano do grupo).
Cada seed vira um run no MLflow (`bm.mlflow_logging.log_bandit_run`); a média ± desvio
fecha a linha "Thompson Sampling" da tabela da Etapa 4.1 no README.

Prior documentado (exigência do enunciado): este Thompson Sampling é **contextual**
(regressão linear Bayesiana por braço), não o Beta-Bernoulli "clássico" sem contexto que
o plano sugere como opção mais simples. O prior é Normal(0, alpha^2 * I) sobre os
coeficientes de cada braço; alpha=1.0 é o prior não-informativo (mesma ideia do
Beta(1,1): nenhum conhecimento prévio sobre qual braço é melhor).
"""

import statistics
from pathlib import Path

import joblib
import pandas as pd

from bm.data_prep import CONTEXT_FEATURES, arm_stats
from bm.evaluation import summarize_policy
from bm.experiments.train import train_bandit
from bm.mlflow_logging import log_bandit_run
from bm.models.bandit import ThompsonSampling

SEEDS: list[int] = list(range(10))
ALPHA_PRIOR: float = 1.0


def _n_features(frame: pd.DataFrame, preprocessor) -> int:
    return preprocessor.transform(frame[CONTEXT_FEATURES].head(1)).shape[1]


def run(frame: pd.DataFrame, preprocessor, seeds: list[int] = SEEDS):
    """Treina uma rodada de Thompson por seed e loga cada uma no MLflow.

    Devolve `(resultados, ultimo_bandit)` -- `resultados` é uma linha por seed
    (conversão, regret, n_amostras, run_id), `ultimo_bandit` é o estado treinado na
    última seed (usado pra salvar `models/thompson.joblib` e gerar recomendações do
    golden set).
    """
    p_oracle = arm_stats(frame)["conversao"].max()
    n_features = _n_features(frame, preprocessor)

    resultados = []
    bandit = None
    for seed in seeds:
        bandit = ThompsonSampling(alpha=ALPHA_PRIOR, n_features=n_features, seed=seed)
        bandit, rewards_matched = train_bandit(
            bandit, frame, preprocessor, log_to_mlflow=False
        )
        resumo = summarize_policy(rewards_matched, p_oracle)

        run_id = log_bandit_run(
            run_name=f"thompson_seed{seed}",
            params={
                "policy": "thompson_linear_contextual",
                "prior": "Normal(0, alpha^2 * I) por braco -- nao-informativo",
                "alpha_prior": ALPHA_PRIOR,
                "n_arms": frame["arm"].nunique(),
                "n_features": n_features,
                "seed": seed,
            },
            metrics=resumo,
        )
        resultados.append({"seed": seed, "run_id": run_id, **resumo})
        print(f"seed={seed}: conversao={resumo['conversao']*100:.2f}% "
              f"regret_final={resumo['regret_final']:.1f} n={resumo['n_amostras']}")

    return resultados, bandit


def summarize_across_seeds(resultados: list[dict]) -> dict:
    """Média ± desvio-padrão sobre as seeds -- formato que o plano do grupo exige."""
    return {
        "conversao_media": statistics.mean(r["conversao"] for r in resultados),
        "conversao_desvio": statistics.pstdev(r["conversao"] for r in resultados),
        "regret_media": statistics.mean(r["regret_final"] for r in resultados),
        "regret_desvio": statistics.pstdev(r["regret_final"] for r in resultados),
        "n_amostras_media": statistics.mean(r["n_amostras"] for r in resultados),
    }


if __name__ == "__main__":
    frame_real = pd.read_parquet("data/processed/bandit_frame.parquet")
    preprocessor_real = joblib.load("models/preprocessor.joblib")

    resultados_real, ultimo_bandit = run(frame_real, preprocessor_real)
    agregado = summarize_across_seeds(resultados_real)

    print(f"\nThompson Sampling -- {len(resultados_real)} seeds")
    print(
        f"  conversao: {agregado['conversao_media'] * 100:.2f}% "
        f"+/- {agregado['conversao_desvio'] * 100:.2f}pp"
    )
    print(
        f"  regret final: {agregado['regret_media']:.1f} "
        f"+/- {agregado['regret_desvio']:.1f}"
    )
    print(f"  n amostras (media): {agregado['n_amostras_media']:.0f}")

    model_path = Path("models/thompson.joblib")
    joblib.dump(ultimo_bandit, model_path)
    print(f"\nmodelo salvo em {model_path} (estado da ultima seed rodada)")
