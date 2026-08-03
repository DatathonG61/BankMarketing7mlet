"""Métricas de avaliação da tabela da Etapa 4.1 (Bertelli).

Recebe as recompensas já observadas numa simulação de replay (rodada por `train_bandit`
na Etapa 3, ou por uma versão provisória própria) e calcula regret e conversão — as
colunas da tabela final ("Política | Conversão | Regret acumulado | N amostras usadas")
que o plano do grupo pede. Não depende de nenhuma etapa de outra pessoa: só de uma lista
de 0/1 e da taxa de conversão do braço oráculo (melhor braço histórico, ~14,7%).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from bm.data_prep import arm_stats


def compute_regret(rewards: Sequence[int], p_oracle: float) -> list[float]:
    """Regret acumulado a cada rodada: quanto se perdeu comparado ao braço ótimo.

    `regret_t = t * p_oracle - soma_das_recompensas_ate_t` (fórmula do plano do grupo).
    O último valor da lista é o "regret final" que vai na tabela e no MLflow.
    """
    valores = np.asarray(rewards, dtype=float)
    acumulado = np.cumsum(valores)
    t = np.arange(1, len(valores) + 1)
    return (t * p_oracle - acumulado).tolist()


def summarize_policy(rewards: Sequence[int], p_oracle: float) -> dict[str, float]:
    """Resume uma política (regra fixa, Epsilon-Greedy, Thompson...) numa linha da tabela."""
    n = len(rewards)
    if n == 0:
        return {"conversao": float("nan"), "regret_final": float("nan"), "n_amostras": 0}

    regret = compute_regret(rewards, p_oracle)
    return {
        "conversao": float(np.mean(rewards)),
        "regret_final": float(regret[-1]),
        "n_amostras": n,
    }


def build_metrics_table(
    frame: pd.DataFrame, thompson_seeds_summary: dict[str, float] | None = None
) -> pd.DataFrame:
    """Monta a tabela da Etapa 4.1 (Política | Conversão | Regret acumulado | N amostras usadas).

    As duas primeiras linhas (baselines) vêm direto do histórico via `arm_stats` — não
    dependem de simulação nenhuma. Epsilon-Greedy fica "pendente" até o Adryen
    implementar a classe (Etapa 3). Thompson Sampling fica "pendente" até alguém passar
    `thompson_seeds_summary` (saída de
    `bm.experiments.run_thompson_replay.summarize_across_seeds`, média ± desvio sobre
    múltiplas seeds).
    """
    stats = arm_stats(frame).to_dict("index")

    if thompson_seeds_summary is None:
        thompson_row = {
            "Política": "Thompson Sampling",
            "Conversão (replay)": "pendente",
            "Regret acumulado": "pendente",
            "N amostras usadas": "pendente",
        }
    else:
        s = thompson_seeds_summary
        thompson_row = {
            "Política": "Thompson Sampling (média de 10 seeds)",
            "Conversão (replay)": (
                f"{s['conversao_media'] * 100:.2f}% +/- {s['conversao_desvio'] * 100:.2f}pp"
            ),
            "Regret acumulado": f"{s['regret_media']:.1f} +/- {s['regret_desvio']:.1f}",
            "N amostras usadas": f"{s['n_amostras_media']:.0f}",
        }

    linhas = [
        {
            "Política": "Regra fixa (telephone)",
            "Conversão (replay)": f"{stats['telephone']['conversao_pct']}%",
            "Regret acumulado": "alto",
            "N amostras usadas": stats["telephone"]["n"],
        },
        {
            "Política": "Melhor braço histórico (cellular)",
            "Conversão (replay)": f"{stats['cellular']['conversao_pct']}%",
            "Regret acumulado": "0 (oráculo)",
            "N amostras usadas": stats["cellular"]["n"],
        },
        {
            "Política": "Epsilon-Greedy (epsilon=0.1)",
            "Conversão (replay)": "pendente",
            "Regret acumulado": "pendente",
            "N amostras usadas": "pendente",
        },
        thompson_row,
    ]
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    import sys

    import joblib

    from bm.experiments.run_thompson_replay import run, summarize_across_seeds

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    frame_real = pd.read_parquet("data/processed/bandit_frame.parquet")
    preprocessor_real = joblib.load("models/preprocessor.joblib")

    resultados_real, _ = run(frame_real, preprocessor_real)
    agregado_real = summarize_across_seeds(resultados_real)

    print(build_metrics_table(frame_real, agregado_real).to_string(index=False))
