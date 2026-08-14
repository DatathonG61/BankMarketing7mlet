"""Serviço FastAPI (Etapa 5 — Matheus).

Serve o Thompson Sampling contextual treinado pelo Adryen (`bm.models.bandit.ThompsonSampling`):

- `POST /recommend` — recebe as features do cliente, devolve o braço escolhido por
  `select_arm()` (decisão real, com exploração) e o score estimado de cada braço pela média
  da posterior (`a_inv @ b`, determinístico — mesma lógica de `bm.golden_set`), útil para
  interpretar a decisão sem depender do sorteio.
- `POST /feedback` — recebe o mesmo contexto do cliente + `arm` + `reward` (0/1) e chama
  `bandit.update()`. É este endpoint que fecha o loop online: sem ele o sistema seria um
  modelo batch qualquer, congelado até o próximo retreino.
- `GET  /health` — status do serviço e se o estado do bandit está persistido em disco.

Contratos de entrada (Etapa 2, `bm.data_prep`):

- `models/preprocessor.joblib` — encoder ajustado no treino, carregado (nunca recriado) para
  evitar train/serving skew.
- `bm.data_prep.CONTEXT_FEATURES` — as 18 colunas de contexto, viram 59 pós-encoding.

Estado do bandit: `models/bandit_state.json`, carregado no `lifespan` e regravado a cada
`/feedback` — sem isso a API "esqueceria" tudo a cada restart. Na primeira execução (sem esse
arquivo ainda) parte do `models/thompson.joblib` já treinado pelo Adryen no replay.

`ArmState` (matriz `a`, inversa `a_inv` e vetor `b` da regressão bayesiana por braço) não é
serializável em JSON por padrão — os arrays viram listas na gravação e voltam a `np.ndarray`
na leitura. A seed do gerador aleatório usada em `select_arm` propositalmente não é
persistida: afeta só a exploração de uma chamada, não a crença aprendida.

Executar: `uv run uvicorn bm.api:app --reload`
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from bm.data_prep import ARMS, CONTEXT_FEATURES
from bm.models.bandit import ArmState, ThompsonSampling

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
MODELS_DIR: Path = REPO_ROOT / "models"
PREPROCESSOR_PATH: Path = MODELS_DIR / "preprocessor.joblib"
PRETRAINED_BANDIT_PATH: Path = MODELS_DIR / "thompson.joblib"
BANDIT_STATE_PATH: Path = MODELS_DIR / "bandit_state.json"

API_VERSION: str = "0.1.0"
POLICY_NAME: str = "thompson"


class CustomerFeatures(BaseModel):
    """As 18 features de contexto (`bm.data_prep.CONTEXT_FEATURES`). `contact` fica de fora
    de propósito: é a ação que o bandit escolhe, não um atributo do cliente."""

    model_config = ConfigDict(populate_by_name=True)

    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    month: str
    day_of_week: str
    poutcome: str
    age: float
    campaign: float
    previous: float
    emp_var_rate: float = Field(alias="emp.var.rate")
    cons_price_idx: float = Field(alias="cons.price.idx")
    cons_conf_idx: float = Field(alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(alias="nr.employed")
    was_contacted_before: bool

    def to_frame(self) -> pd.DataFrame:
        data = self.model_dump(by_alias=True)
        return pd.DataFrame([data])[CONTEXT_FEATURES]


class RecommendResponse(BaseModel):
    arm: str
    arm_scores: dict[str, float]
    policy: str = POLICY_NAME


class FeedbackRequest(BaseModel):
    customer: CustomerFeatures
    arm: str
    reward: Literal[0, 1]


class FeedbackResponse(BaseModel):
    status: str
    arm: str
    reward: int


class HealthResponse(BaseModel):
    status: str
    policy: str
    arms: list[str]
    n_features: int
    bandit_state_persisted: bool


def _arm_state_to_json(state: ArmState) -> dict[str, Any]:
    return {"a": state.a.tolist(), "a_inv": state.a_inv.tolist(), "b": state.b.tolist()}


def _arm_state_from_json(data: dict[str, Any]) -> ArmState:
    return ArmState(
        a=np.array(data["a"]), a_inv=np.array(data["a_inv"]), b=np.array(data["b"])
    )


def save_bandit_state(bandit: ThompsonSampling, path: Path = BANDIT_STATE_PATH) -> None:
    """Grava a crença aprendida (`a`, `a_inv`, `b` de cada braço) em disco, como JSON."""
    payload = {
        "alpha": bandit.alpha,
        "n_features": bandit.n_features,
        "arms": {arm: _arm_state_to_json(state) for arm, state in bandit.arms.items()},
    }
    path.write_text(json.dumps(payload))


def load_bandit_state(path: Path = BANDIT_STATE_PATH) -> ThompsonSampling:
    payload = json.loads(path.read_text())
    bandit = ThompsonSampling(alpha=payload["alpha"], n_features=payload["n_features"])
    bandit.arms = {
        arm: _arm_state_from_json(state) for arm, state in payload["arms"].items()
    }
    return bandit


def load_or_init_bandit() -> ThompsonSampling:
    """Estado persistido tem prioridade; na primeira execução parte do modelo já treinado
    pelo Adryen no replay (`models/thompson.joblib`) e grava o estado inicial."""
    if BANDIT_STATE_PATH.exists():
        return load_bandit_state()
    if PRETRAINED_BANDIT_PATH.exists():
        bandit = joblib.load(PRETRAINED_BANDIT_PATH)
        save_bandit_state(bandit)
        return bandit
    raise FileNotFoundError(
        f"nenhum bandit disponivel: nem {BANDIT_STATE_PATH} nem {PRETRAINED_BANDIT_PATH} existem "
        "(rode a Etapa 3 antes de subir a API)"
    )


def _posterior_mean_scores(bandit: ThompsonSampling, x: np.ndarray) -> dict[str, float]:
    """Score por braço pela média da posterior (sem sortear) — mesma lógica de
    `bm.golden_set.recommend_for_golden_set`. Não é uma probabilidade calibrada, serve para
    comparar os braços entre si."""
    return {arm: float((state.a_inv @ state.b) @ x) for arm, state in bandit.arms.items()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not PREPROCESSOR_PATH.exists():
        raise RuntimeError(
            f"preprocessor nao encontrado em {PREPROCESSOR_PATH} - "
            "rode `uv run python -m src.data_prep` (Etapa 2) antes de subir a API"
        )
    app.state.preprocessor = joblib.load(PREPROCESSOR_PATH)
    app.state.bandit = load_or_init_bandit()
    yield


app = FastAPI(
    title="Bank Marketing Bandit API",
    version=API_VERSION,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    bandit: ThompsonSampling = app.state.bandit
    return HealthResponse(
        status="ok",
        policy=POLICY_NAME,
        arms=sorted(bandit.arms.keys()),
        n_features=bandit.n_features,
        bandit_state_persisted=BANDIT_STATE_PATH.exists(),
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(customer: CustomerFeatures) -> RecommendResponse:
    bandit: ThompsonSampling = app.state.bandit
    x = app.state.preprocessor.transform(customer.to_frame())[0]

    chosen_arm = bandit.select_arm(x)
    return RecommendResponse(arm=chosen_arm, arm_scores=_posterior_mean_scores(bandit, x))


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(payload: FeedbackRequest) -> FeedbackResponse:
    if payload.arm not in ARMS:
        raise HTTPException(
            status_code=422, detail=f"arm invalido: {payload.arm!r}, esperado um de {ARMS}"
        )

    bandit: ThompsonSampling = app.state.bandit
    x = app.state.preprocessor.transform(payload.customer.to_frame())[0]

    bandit.update(payload.arm, x, payload.reward)
    save_bandit_state(bandit)

    return FeedbackResponse(status="updated", arm=payload.arm, reward=payload.reward)
