"""Roda o Epsilon-Greedy em múltiplas seeds -- Etapa 4.1 e 7.1 (Bertelli).

Espelha `run_thompson_replay.py`: mesmo `train_bandit` (Etapa 3), mesma ideia de várias
seeds com média ± desvio (bandit é estocástico, plano do grupo exige reportar variância).
`EpsilonGreedy` é a variante clássica, não-contextual (`bm.models.bandit.EpsilonGreedy`)
-- serve de contraponto mais simples ao Thompson contextual do Adryen na tabela da
Etapa 4.1.
"""

import statistics

import joblib
import pandas as pd

from bm.data_prep import arm_stats
from bm.evaluation import summarize_policy
from bm.experiments.train import train_bandit
from bm.mlflow_logging import log_bandit_run
from bm.models.bandit import EpsilonGreedy

SEEDS: list[int] = list(range(10))
EPSILON: float = 0.1


def run(frame: pd.DataFrame, preprocessor, seeds: list[int] = SEEDS, epsilon: float = EPSILON):
    """Treina uma rodada de Epsilon-Greedy por seed e loga cada uma no MLflow.

    Devolve `(resultados, ultimo_bandit)` -- mesmo formato de `run_thompson_replay.run`,
    consumido por `summarize_across_seeds` e por `bm.evaluation.build_metrics_table`.
    """
    p_oracle = arm_stats(frame)["conversao"].max()

    resultados = []
    bandit = None
    for seed in seeds:
        bandit = EpsilonGreedy(epsilon=epsilon, seed=seed)
        bandit, rewards_matched = train_bandit(
            bandit, frame, preprocessor, log_to_mlflow=False
        )
        resumo = summarize_policy(rewards_matched, p_oracle)

        run_id = log_bandit_run(
            run_name=f"epsilon_greedy_seed{seed}",
            params={
                "policy": "epsilon_greedy",
                "epsilon": epsilon,
                "n_arms": frame["arm"].nunique(),
                "seed": seed,
            },
            metrics=resumo,
        )
        resultados.append({"seed": seed, "run_id": run_id, **resumo})
        print(f"seed={seed}: conversao={resumo['conversao']*100:.2f}% "
              f"regret_final={resumo['regret_final']:.1f} n={resumo['n_amostras']}")

    return resultados, bandit


def summarize_across_seeds(resultados: list[dict]) -> dict:
    """Mesmo formato de `run_thompson_replay.summarize_across_seeds` -- média ± desvio."""
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

    resultados_real, _ = run(frame_real, preprocessor_real)
    agregado = summarize_across_seeds(resultados_real)

    print(f"\nEpsilon-Greedy (epsilon={EPSILON}) -- {len(resultados_real)} seeds")
    print(
        f"  conversao: {agregado['conversao_media'] * 100:.2f}% "
        f"+/- {agregado['conversao_desvio'] * 100:.2f}pp"
    )
    print(
        f"  regret final: {agregado['regret_media']:.1f} "
        f"+/- {agregado['regret_desvio']:.1f}"
    )
    print(f"  n amostras (media): {agregado['n_amostras_media']:.0f}")
