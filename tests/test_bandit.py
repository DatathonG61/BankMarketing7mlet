"""Testes do que já existe em `src/bm/models/bandit.py` — Etapa 4.3 (Bertelli).

`EpsilonGreedy` e `BaselineBasic` ainda são esqueletos do Adryen (Etapa 3); estes testes
cobrem só o `ThompsonSampling`, que já está implementado. O último teste é o "cenário
sintético" que o plano do grupo pede explicitamente nesta etapa: braços com taxa de
conversão conhecida, o bandit deve aprender a preferir o melhor.

Quando o Adryen terminar o `EpsilonGreedy`, os dois primeiros testes servem de modelo pra
cobrir a classe nova também (mesma interface `select_arm`/`update`).
"""

import numpy as np

from bm.models.bandit import ThompsonSampling

VALID_ARMS = {"cellular", "telephone"}


def test_select_arm_returns_valid_arm():
    """Confere que `select_arm` sempre devolve um braço válido, nunca outro valor."""
    bandit = ThompsonSampling(alpha=1.0, n_features=3, seed=0)
    customer = np.array([1.0, 0.0, 1.0])
    assert bandit.select_arm(customer) in VALID_ARMS


def test_update_changes_state():
    """Usa 'cellular' direto (não o que `select_arm` devolveu) — assim, se o dicionário
    interno de braços for criado com o nome errado, este teste falha com KeyError em vez
    de silenciosamente aceitar qualquer nome que o bandit escolher."""
    bandit = ThompsonSampling(alpha=1.0, n_features=3, seed=0)
    customer = np.array([1.0, 0.0, 1.0])
    b_antes = bandit.arms["cellular"].b.copy()
    bandit.update("cellular", customer, reward=1)
    assert not np.array_equal(bandit.arms["cellular"].b, b_antes)


def test_converges_on_synthetic_arms():
    """Braço 'cellular' converte 80% das vezes, 'telephone' só 10% — depois de várias
    rodadas o bandit deve escolher 'cellular' na grande maioria das vezes."""
    rng = np.random.default_rng(42)
    true_rate = {"cellular": 0.8, "telephone": 0.1}
    customer = np.array([1.0])

    bandit = ThompsonSampling(alpha=1.0, n_features=1, seed=42)
    choices = []
    for _ in range(400):
        arm = bandit.select_arm(customer)
        reward = int(rng.random() < true_rate[arm])
        bandit.update(arm, customer, reward)
        choices.append(arm)

    ultimas_100 = choices[-100:]
    taxa_melhor_braco = ultimas_100.count("cellular") / len(ultimas_100)
    assert taxa_melhor_braco > 0.7
