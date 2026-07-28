"""Algoritmos de bandit (Etapa 3 — Adryen).

Esqueleto da Etapa 0. A implementação é da Etapa 3 e não deve ser preenchida aqui sem combinar
com o dono da etapa.

`EpsilonGreedy` e `ThompsonSampling` devem expor a **mesma** interface `select_arm()` / `update()`
— é a interface que a API da Etapa 5 consome em `/recommend` e `/feedback`.

**Priors (exigência explícita do enunciado):** o Thompson Sampling usa Beta(alpha=1, beta=1) por
padrão. Beta(1,1) é a distribuição **uniforme** em [0,1] — o prior não-informativo: antes de
observar qualquer dado, toda taxa de conversão é igualmente plausível. Não embutimos conhecimento
prévio sobre qual canal é melhor; o bandit precisa descobrir sozinho.

**Avaliação por replay / rejection sampling** (Li et al., 2011): percorrer `bandit_frame.parquet`
na ordem cronológica original; a cada linha o bandit escolhe um braço e **só contabiliza a
recompensa e atualiza o estado quando a escolha coincide com o braço de fato usado no histórico**.

Dados de entrada (contrato da Etapa 2, ver `src/data_prep.py`):

- `data/processed/bandit_frame.parquet` — colunas `arm`, `reward` + 18 features de contexto.
- Braços: `cellular` / `telephone` (2). Recompensa: 0/1. Taxa-base real: 11,27%.
- Baselines a superar: regra fixa ingênua (~5,2%) e melhor braço histórico (~14,7%).

Instrumentar com MLflow via `src.tracking.setup_mlflow("datathon-bandit")`, logando priors,
epsilon, seed e as métricas (conversão no replay, regret, n_matched). Reportar **média ± desvio
sobre múltiplas seeds** — bandits são estocásticos, uma rodada única não é resultado.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

class BaselineBasic:
    def __init__(self):
        return
    
    def select_arm(self, customer):
        return "cellphone"
    
    def update(self, arm, customer, reward):
        return

class EpsilonGreedy:
    def __init__(self):
        return
    
    def select_arm(self, customer):
        return
    
    def update(self, arm, customer, reward):
        return
    
@dataclass
class ArmState:
    a: np.ndarray
    b: np.ndarray
    a_inv: np.ndarray
    
class ThompsonSampling:
    def __init__(self, alpha: int, n_features: int, seed: int | None = None):
        self.alpha = alpha
        self.n_features = n_features
        self.rng = np.random.default_rng(seed)

        arms = {"cellular", "telephone"}
        self.arms = {}
        for arm in arms:
            self.arms[arm] = ArmState(
                a=np.eye(n_features),
                a_inv=np.eye(n_features),
                b=np.zeros(n_features)
            )

    def select_arm(self, customer):
        x = np.asarray(customer, dtype=float)

        best_arm = None
        best_score = -np.inf

        for arm, state in self.arms.items():
            a_inv = state.a_inv
            mu = a_inv @ state.b
            cov = (self.alpha ** 2) * a_inv

            #try:
            #    theta = self.rng.multivariate_normal(mu, cov)
            #except np.linalg.LinAlgError:
                #print("Erro ao gerar theta")
                #print("np.linalg.cond(cov) = ", np.linalg.cond(cov))
                #print("Menor autovalor:", np.linalg.eigvalsh(cov).min())
                #print("Maior autovalor:", np.linalg.eigvalsh(cov).max())
                #print("Tem NaN:", np.isnan(cov).any())
                #print("Tem Inf:", np.isinf(cov).any())
                #print("Shape:", cov.shape)
                #print("All close = " ,np.allclose(cov, cov.T))
                #print("np.max(np.abs(cov - cov.T)) = ", np.max(np.abs(cov - cov.T)))
                #raise

            L = np.linalg.cholesky(cov)
            z = self.rng.standard_normal(len(mu))
            theta = mu + L @ z

            score = theta @ x

            if score > best_score:
                best_score = score
                best_arm = arm

        return best_arm
    
    def update(self, arm, customer, reward):
        x = np.asarray(customer, dtype=float)
        state = self.arms[arm]
        state.a += np.outer(x, x)

        a_inv = state.a_inv
        aX = a_inv @ x
        denominator = 1.0 + x @ aX
        state.a_inv = a_inv - np.outer(aX, aX) / denominator

        state.b += reward * x
