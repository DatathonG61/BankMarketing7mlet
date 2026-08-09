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


def _seeds_summary_row(politica: str, summary: dict[str, float] | None) -> dict[str, str]:
    """Formata a linha de uma política simulada em várias seeds (Thompson ou Epsilon-Greedy).

    `None` vira "pendente" -- útil quando quem chama `build_metrics_table` ainda não
    rodou a simulação daquela política.
    """
    if summary is None:
        return {
            "Política": politica,
            "Conversão (replay)": "pendente",
            "Regret acumulado": "pendente",
            "N amostras usadas": "pendente",
        }
    s = summary
    return {
        "Política": f"{politica} (média de 10 seeds)",
        "Conversão (replay)": (
            f"{s['conversao_media'] * 100:.2f}% +/- {s['conversao_desvio'] * 100:.2f}pp"
        ),
        "Regret acumulado": f"{s['regret_media']:.1f} +/- {s['regret_desvio']:.1f}",
        "N amostras usadas": f"{s['n_amostras_media']:.0f}",
    }


def build_metrics_table(
    frame: pd.DataFrame,
    thompson_seeds_summary: dict[str, float] | None = None,
    epsilon_seeds_summary: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Monta a tabela da Etapa 4.1 (Política | Conversão | Regret acumulado | N amostras usadas).

    As duas primeiras linhas (baselines) vêm direto do histórico via `arm_stats` — não
    dependem de simulação nenhuma. `thompson_seeds_summary`/`epsilon_seeds_summary` vêm
    de `bm.experiments.run_thompson_replay.summarize_across_seeds` /
    `bm.experiments.run_epsilon_replay.summarize_across_seeds` (média ± desvio sobre
    múltiplas seeds); ficam "pendente" se não forem passados.
    """
    stats = arm_stats(frame).to_dict("index")

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
        _seeds_summary_row("Epsilon-Greedy (epsilon=0.1)", epsilon_seeds_summary),
        _seeds_summary_row("Thompson Sampling", thompson_seeds_summary),
    ]
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    import sys

    import joblib

    from bm.experiments.run_epsilon_replay import run as run_epsilon
    from bm.experiments.run_epsilon_replay import summarize_across_seeds as summarize_epsilon
    from bm.experiments.run_thompson_replay import run as run_thompson
    from bm.experiments.run_thompson_replay import summarize_across_seeds as summarize_thompson

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    frame_real = pd.read_parquet("data/processed/bandit_frame.parquet")
    preprocessor_real = joblib.load("models/preprocessor.joblib")

    resultados_thompson, _ = run_thompson(frame_real, preprocessor_real)
    agregado_thompson = summarize_thompson(resultados_thompson)

    resultados_epsilon, _ = run_epsilon(frame_real, preprocessor_real)
    agregado_epsilon = summarize_epsilon(resultados_epsilon)

    tabela = build_metrics_table(frame_real, agregado_thompson, agregado_epsilon)
    print(tabela.to_string(index=False))
