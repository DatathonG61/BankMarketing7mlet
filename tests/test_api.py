"""Testes da API (Etapa 4.3 itens d/e — Bertelli), sobre o que o Matheus implementou na
Etapa 5 (`src/bm/api.py`).

Usa `TestClient` como context manager pra disparar o `lifespan` (carrega
`models/preprocessor.joblib` e `models/thompson.joblib`, já commitados no repo). Não testa
`/feedback` com payload válido de propósito: isso gravaria `models/bandit_state.json` no
disco de verdade a cada `pytest -q` — o caso de payload inválido (422) já cobre a validação
sem esse efeito colateral, e é o único cenário de `/feedback` que o plano pede nesta etapa.
"""

from fastapi.testclient import TestClient

from bm.api import app

CUSTOMER_PAYLOAD = {
    "job": "admin.",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "month": "may",
    "day_of_week": "mon",
    "poutcome": "nonexistent",
    "age": 35,
    "campaign": 1,
    "previous": 0,
    "emp.var.rate": 1.1,
    "cons.price.idx": 93.9,
    "cons.conf.idx": -36.4,
    "euribor3m": 4.86,
    "nr.employed": 5191.0,
    "was_contacted_before": False,
}


def test_health_returns_200():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert set(body["arms"]) == {"cellular", "telephone"}


def test_recommend_with_valid_payload_returns_200():
    with TestClient(app) as client:
        response = client.post("/recommend", json=CUSTOMER_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        assert body["arm"] in {"cellular", "telephone"}
        assert set(body["arm_scores"]) == {"cellular", "telephone"}
        assert body["policy"] == "thompson"


def test_recommend_with_missing_field_returns_422():
    """Payload inválido (falta `job`, campo obrigatório) — a API deve rejeitar, não
    devolver 200 com um cliente incompleto."""
    payload_incompleto = {k: v for k, v in CUSTOMER_PAYLOAD.items() if k != "job"}
    with TestClient(app) as client:
        response = client.post("/recommend", json=payload_incompleto)
        assert response.status_code == 422


def test_feedback_with_invalid_arm_returns_422():
    with TestClient(app) as client:
        response = client.post(
            "/feedback",
            json={"customer": CUSTOMER_PAYLOAD, "arm": "carrier_pigeon", "reward": 1},
        )
        assert response.status_code == 422
