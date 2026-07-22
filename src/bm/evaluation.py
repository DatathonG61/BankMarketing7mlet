"""Métricas de avaliação da tabela da Etapa 4.1 (Bertelli).

Recebe as recompensas já observadas numa simulação de replay (rodada por `train_bandit`
na Etapa 3, ou por uma versão provisória própria) e calcula regret e conversão — as
colunas da tabela final ("Política | Conversão | Regret acumulado | N amostras usadas")
que o plano do grupo pede. Não depende de nenhuma etapa de outra pessoa: só de uma lista
de 0/1 e da taxa de conversão do braço oráculo (melhor braço histórico, ~14,7%).
"""

from collections.abc import Sequence

import numpy as np


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
