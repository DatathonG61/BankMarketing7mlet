"""Testes da análise de exploração — Etapa 3.4.

O risco real deste módulo não é o desenho do gráfico: é o `trace` de `train_bandit`
mentir. Se ele registrar só as linhas aproveitadas, ou alterar a simulação, as figuras
passam a contar uma história diferente da tabela de métricas da Etapa 4.1 — e ninguém
percebe olhando o PNG. Estes testes travam exatamente isso.
"""

import numpy as np
import pandas as pd
import pytest

from bm.data_prep import CONTEXT_FEATURES
from bm.exploration import _matched_rewards, _matched_steps, resumo_exploracao
from bm.experiments.train import train_bandit
from bm.models.bandit import EpsilonGreedy


class _PreprocessorFake:
    """Devolve uma matriz numérica com uma coluna por feature de contexto."""

    def transform(self, df):
        return np.ones((len(df), 4), dtype=float)


def _frame(n=200):
    metade = n // 2
    dados = {
        "arm": ["cellular"] * metade + ["telephone"] * (n - metade),
        "reward": [1, 0] * (n // 2),
    }
    for coluna in CONTEXT_FEATURES:
        dados[coluna] = [0] * n
    return pd.DataFrame(dados)


def test_trace_registra_todas_as_rodadas_nao_so_as_aproveitadas():
    """O trace precisa cobrir a base inteira: é nas linhas descartadas pelo rejection
    sampling que aparece o quanto a política ainda está explorando."""
    frame = _frame()
    trace: list[tuple] = []
    _, rewards_matched = train_bandit(
        EpsilonGreedy(epsilon=0.1, seed=0), frame, _PreprocessorFake(),
        log_to_mlflow=False, trace=trace,
    )

    assert len(trace) == len(frame)
    assert len(rewards_matched) < len(trace), "algum descarte tem que ter acontecido"
    assert len(_matched_rewards(trace)) == len(rewards_matched)


def test_trace_nao_altera_a_simulacao():
    """Mesma seed, com e sem trace, tem que produzir exatamente o mesmo resultado —
    senão as figuras da Etapa 3.4 e a tabela da Etapa 4.1 divergem em silêncio."""
    frame = _frame()

    _, sem_trace = train_bandit(
        EpsilonGreedy(epsilon=0.1, seed=7), frame, _PreprocessorFake(), log_to_mlflow=False
    )
    trace: list[tuple] = []
    _, com_trace = train_bandit(
        EpsilonGreedy(epsilon=0.1, seed=7), frame, _PreprocessorFake(),
        log_to_mlflow=False, trace=trace,
    )

    assert sem_trace == com_trace


def test_recompensa_do_trace_bate_com_o_historico():
    """Nas linhas aproveitadas, a recompensa registrada tem que ser a da própria linha."""
    frame = _frame()
    trace: list[tuple] = []
    train_bandit(
        EpsilonGreedy(epsilon=0.1, seed=1), frame, _PreprocessorFake(),
        log_to_mlflow=False, trace=trace,
    )

    for step, braco, casou, recompensa in trace:
        if casou:
            assert braco == frame["arm"].iloc[step]
            assert recompensa == frame["reward"].iloc[step]
        else:
            assert recompensa == -1, "linha descartada não tem desfecho observável"


def test_steps_aproveitados_sao_crescentes():
    """As curvas interpolam sobre `_matched_steps`; `np.interp` exige eixo crescente."""
    frame = _frame()
    trace: list[tuple] = []
    train_bandit(
        EpsilonGreedy(epsilon=0.1, seed=3), frame, _PreprocessorFake(),
        log_to_mlflow=False, trace=trace,
    )
    steps = _matched_steps(trace)
    assert np.all(np.diff(steps) > 0)


def test_resumo_exploracao_tem_uma_linha_por_politica():
    frame = _frame()
    traces = {}
    for politica in ("Thompson Sampling", "Epsilon-Greedy"):
        trace: list[tuple] = []
        train_bandit(
            EpsilonGreedy(epsilon=0.1, seed=0), frame, _PreprocessorFake(),
            log_to_mlflow=False, trace=trace,
        )
        traces[politica] = [trace]

    tabela = resumo_exploracao(traces)
    assert len(tabela) == 2
    assert list(tabela["Política"]) == ["Thompson Sampling", "Epsilon-Greedy"]


@pytest.mark.parametrize("figura", [
    "conversao_acumulada.png",
    "escolha_por_braco.png",
    "regret_acumulado.png",
    "posterior_thompson.png",
])
def test_figuras_estao_versionadas(figura):
    """As figuras são entregável: o README aponta pra elas e a banca abre pelo GitHub.
    Se alguém apagar uma, este teste avisa antes da apresentação."""
    from pathlib import Path

    caminho = Path("reports/figures") / figura
    assert caminho.exists(), f"rode `uv run python -m bm.exploration` para gerar {figura}"
    assert caminho.stat().st_size > 10_000, "figura truncada ou vazia"
