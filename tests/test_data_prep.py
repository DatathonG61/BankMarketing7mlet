"""Testes da Etapa 2 (`src/bm/data_prep.py`) — Etapa 4.3 (Bertelli).

Cobre o pipeline de limpeza e o contrato `bandit_frame` que a Etapa 3 (bandit), a Etapa 4
(golden set) e a Etapa 5 (API) consomem. Usa uma base sintética pequena construída em
memória — não depende do CSV real nem de nenhuma etapa de outra pessoa estar pronta.
"""

import pandas as pd

from bm.data_prep import ARMS, CONTEXT_FEATURES, build_bandit_frame, clean


def _raw_rows() -> pd.DataFrame:
    """3 linhas cruas mínimas, com todas as colunas que `clean`/`build_bandit_frame` esperam."""
    base = {
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
        "cons.price.idx": 93.2,
        "cons.conf.idx": -36.4,
        "euribor3m": 4.9,
        "nr.employed": 5191.0,
        "duration": 120,
    }
    rows = [
        {**base, "contact": "cellular", "y": "yes", "pdays": 999},
        {**base, "contact": "telephone", "y": "no", "pdays": 3},
        {**base, "contact": "cellular", "y": "no", "pdays": 999},
    ]
    return pd.DataFrame(rows)


def test_clean_removes_duration():
    out = clean(_raw_rows())
    assert "duration" not in out.columns


def test_clean_removes_duplicates():
    raw_com_duplicata = pd.concat([_raw_rows(), _raw_rows().iloc[[0]]], ignore_index=True)
    out = clean(raw_com_duplicata)
    assert len(out) == len(_raw_rows())


def test_clean_creates_was_contacted_before():
    out = clean(_raw_rows())
    assert "pdays" not in out.columns
    assert out["was_contacted_before"].tolist() == [False, True, False]


def test_build_bandit_frame_has_expected_columns():
    frame = build_bandit_frame(clean(_raw_rows()))
    expected = {"arm", "reward", *CONTEXT_FEATURES}
    assert expected.issubset(frame.columns)


def test_build_bandit_frame_reward_is_binary():
    frame = build_bandit_frame(clean(_raw_rows()))
    assert set(frame["reward"].unique()) <= {0, 1}


def test_arm_values_are_valid():
    frame = build_bandit_frame(clean(_raw_rows()))
    assert frame["arm"].isin(ARMS).all()
