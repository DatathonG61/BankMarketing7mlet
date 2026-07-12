"""Serviço FastAPI (Etapa 5 — Matheus).

Esqueleto da Etapa 0. A implementação é da Etapa 5 e não deve ser preenchida aqui sem combinar
com o dono da etapa.

Endpoints a implementar:

- `POST /recommend` — recebe as features do cliente, devolve o braço escolhido e as
  probabilidades estimadas por braço.
  Ex.: `{"arm": "cellular", "arm_probs": {"cellular": 0.14, "telephone": 0.05}, "policy": "thompson"}`
- `POST /feedback`  — recebe `{"arm": ..., "reward": 0|1}` e chama `bandit.update()`.
  É este endpoint que torna o sistema **adaptativo**: o loop de aprendizado online. É o
  diferencial em relação a um modelo batch, cuja decisão fica congelada até o próximo retreino.
- `GET  /health`    — status + versão do modelo.

Contratos de entrada (Etapa 2, ver `src/data_prep.py`):

- `models/preprocessor.joblib` — o encoder **ajustado no treino**. Carregar este artefato (e não
  recriá-lo) é o que evita train/serving skew: ele carrega as médias/desvios e as categorias
  exatas vistas no treino.
- `src.data_prep.CONTEXT_FEATURES` — as 18 colunas de contexto esperadas (viram 59 pós-encoding).
- `src/bandit.py` (Etapa 3) — as classes com `select_arm()` / `update()`.

Persistir o estado do bandit em `models/bandit_state.json` e carregá-lo no startup (evento
`lifespan`): sem isso a API "esquece" tudo a cada restart.

Executar: `uv run uvicorn src.api:app --reload`
"""
