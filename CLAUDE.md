# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A POSTECH/MLET datathon project (Gabriel + 3 teammates: Adryen, Bertelli, Matheus). The brief asks
for an **adaptive experimentation platform** (multi-armed bandit), *not* a traditional classifier: given
the Bank Marketing dataset (`henriqueyamahata/bank-marketing`, `bank-additional-full.csv`, 41.188 linhas
× 21 colunas), decide per-client which contact **arm** (channel/day) to use, observe the reward
(`y == "yes"`), and update online — as opposed to a batch model whose decision is frozen until retrain.

Full task spec lives in `ref_doc/` (two PDFs — the official datathon brief and the team's own refined
plan). Read `ref_doc/Plano 7MLET Refinado.pdf` before making structural decisions; it defines the
target architecture, the data contract between stages, and the grading traps below.

Detailed responsibilities per teammate (for context on PR ownership, not enforced by tooling):
- **Gabriel**: repo setup, EDA, data cleaning pipeline (`src/data_prep.py`)
- **Adryen**: bandit algorithms (`src/bandit.py`), baseline + simulation, propensity baseline model
- **Bertelli**: MLflow tracking, evaluation metrics, golden set, automated tests
- **Matheus**: FastAPI service (`src/api.py`), cloud architecture writeup

## Current state vs. target architecture

The repo is early-stage — only Etapa 0/1 work is committed so far (EDA notebook, raw CSV). Target
layout from the plan (not all present yet):

```
data/raw/                       # bank-additional-full.csv (committed)
data/processed/                 # cleaned table + bandit_frame.parquet (Etapa 2 output)
notebooks/01-eda.ipynb          # EDA + leakage analysis (exists)
notebooks/01-baseline.ipynb     # propensity baseline (exists, imports from a `bankmarketing`
                                 # package that does not exist yet under src/ — expect ImportError
                                 # until src/data.py / the package is created)
src/data_prep.py                # load_raw(), clean(), build_bandit_frame() — pure functions,
                                 # the data contract shared by notebooks, API, and tests
src/bandit.py                   # EpsilonGreedy / ThompsonSampling, common select_arm()/update() interface
src/api.py                      # FastAPI: POST /recommend, POST /feedback, GET /health
models/                         # preprocessor.joblib, bandit_state.json
mlruns/                         # MLflow tracking store
```

`src/data_prep.py`, `src/bandit.py`, `src/api.py` are the load-bearing modules once they exist — the
whole point of the architecture is that the API, the bandit simulation, and the golden-set evaluation
all call the *same* cleaning/encoding functions instead of duplicating notebook logic (avoids
train/serving skew). When adding pipeline code, put it there rather than inlining it in a notebook.

## Commands

Dependencies and environment are managed with `uv` (`pyproject.toml` + `uv.lock`, both committed).

```bash
uv sync                                   # install/update the venv from the lockfile
uv run jupyter lab                        # work in notebooks/
uv run pytest -q                          # run tests (tests/ not created yet)
uv run uvicorn src.api:app --reload       # run the API once src/api.py exists
uv add <package>                          # commit pyproject.toml + uv.lock together
```

Requires Python >= 3.13 (see `.python-version`).

## Key domain facts (from EDA, `notebooks/01-eda.ipynb`)

- **Target is imbalanced**: ~11.3% `y == "yes"`. Never use accuracy; use ROC-AUC / PR-AUC / regret.
- **`duration` is leakage and must never be a model feature** — it's only known after the call ends
  (corr 0.40 with y). The plan explicitly calls this "desclassificação moral imediata" if it slips in.
  `notebooks/01-baseline.ipynb` asserts `"duration" not in modeling_table.columns` — keep that
  assertion pattern in any new modeling code.
- **`pdays == 999`** (96% of rows) is a sentinel for "never contacted before", not a numeric value —
  it's converted to a boolean flag (`foi_contatado_antes` / `was_contacted_before`), never used raw.
- **`"unknown"` string values** are disguised missing data (default 21%, education 4%, etc.) — kept as
  their own category rather than imputed (it's informative: "client declined to answer").
- **Macro-economic columns** (`emp.var.rate`, `euribor3m`, `nr.employed`, `cons.price.idx`,
  `cons.conf.idx`) are highly collinear (0.91–0.97) and act as a proxy for the 2008–2010 time period —
  treat as candidates for redundancy/removal, not as independent signal.
- **Row order is chronological (May 2008 → Nov 2010) and must never be shuffled** — the bandit replay
  simulation (Etapa 3) depends on processing rows in original order; shuffling breaks the online
  evaluation logic and destroys the intended "regime drift" narrative.
- **Bandit arms** = contact channel (`contact`: cellular vs telephone), optionally crossed with
  `day_of_week` for more granularity. Reward = `y == "yes"`. Evaluated via **replay/rejection
  sampling** (Li et al., 2011): step through rows in order, only count/update when the bandit's chosen
  arm matches the arm actually used historically.
- Propensity scores for bandit context features are generated **out-of-fold** (`cross_val_predict`)
  to avoid a model leaking its own training rows into the context signal.

## Conventions to preserve

- Never rebalance (no SMOTE/over/undersampling) for the propensity baseline — scores are consumed
  downstream as calibrated probabilities; resampling distorts the true ~11% base rate.
- Log bandit run parameters (priors, epsilon, seed) and metrics via MLflow — priors must be documented
  explicitly (e.g. `Beta(1,1)` = uninformative prior), it's a stated grading requirement.
- Report bandit results across multiple seeds (mean ± std), not a single run — bandits are stochastic.
