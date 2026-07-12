"""Serviço FastAPI (Etapa 5 — Matheus).

Endpoints:

- `POST /recommend` — recebe as features do cliente, devolve o braço escolhido e as
  probabilidades estimadas por braço.
- `POST /feedback`  — recebe `{"arm": ..., "reward": 0|1}` e chama `bandit.update()`.
  É este endpoint que torna o sistema **adaptativo**: o loop de aprendizado online.
- `GET  /health`    — status + versão do modelo.

Consome `models/preprocessor.joblib` (Etapa 2) e as classes de `src/bandit.py` (Etapa 3),
carregando o estado salvo em `models/bandit_state.json` no startup e persistindo a cada
feedback — sem isso a API "esquece" tudo a cada restart.

Executar: `uv run uvicorn src.api:app --reload`
"""

from fastapi import FastAPI

app = FastAPI(
    title="Bank Marketing — Plataforma de Experimentacao Adaptativa",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Status do serviço."""
    return {"status": "ok", "version": app.version}
