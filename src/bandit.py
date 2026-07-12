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
