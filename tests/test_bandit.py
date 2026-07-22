"""Testes do que já existe em `src/bm/bandit.py` — Etapa 4.3 (Bertelli).

`EpsilonGreedy`, `BaselineBasic` e `train_bandit` ainda são esqueletos do Adryen (Etapa 3);
estes testes cobrem só o `ThompsonSampling`, que já está implementado. O último teste é o
"cenário sintético" que o plano do grupo pede explicitamente nesta etapa: braços com taxa
de conversão conhecida, o bandit deve aprender a preferir o melhor.

Quando o Adryen terminar o `EpsilonGreedy`, os dois primeiros testes servem de modelo pra
cobrir a classe nova também (mesma interface `select_arm`/`update`).
"""

import numpy as np

from bm.bandit import ThompsonSampling

VALID_ARMS = {"cellular", "telephone"}


def test_select_arm_returns_valid_arm():
    bandit = ThompsonSampling(n_features=3, seed=0)
    customer = np.array([1.0, 0.0, 1.0])
    assert bandit.select_arm(customer) in VALID_ARMS


def test_update_changes_state():
    bandit = ThompsonSampling(n_features=3, seed=0)
    customer = np.array([1.0, 0.0, 1.0])
    arm = bandit.select_arm(customer)
    b_antes = bandit.arms[arm].b.copy()
    bandit.update(arm, customer, reward=1)
    assert not np.array_equal(bandit.arms[arm].b, b_antes)


def test_converges_on_synthetic_arms():
    """Braço 'cellular' converte 80% das vezes, 'telephone' só 10% — depois de várias
    rodadas o bandit deve escolher 'cellular' na grande maioria das vezes."""
    rng = np.random.default_rng(42)
    true_rate = {"cellular": 0.8, "telephone": 0.1}
    customer = np.array([1.0])

    bandit = ThompsonSampling(n_features=1, seed=42)
    choices = []
    for _ in range(400):
        arm = bandit.select_arm(customer)
        reward = int(rng.random() < true_rate[arm])
        bandit.update(arm, customer, reward)
        choices.append(arm)

    ultimas_100 = choices[-100:]
    taxa_melhor_braco = ultimas_100.count("cellular") / len(ultimas_100)
    assert taxa_melhor_braco > 0.7
