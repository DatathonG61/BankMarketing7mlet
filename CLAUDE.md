# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A POSTECH/MLET datathon project (Gabriel + 3 teammates: Adryen, Bertelli, Matheus). The brief asks
for an **adaptive experimentation platform** (multi-armed bandit), *not* a traditional classifier: given
the Bank Marketing dataset (`henriqueyamahata/bank-marketing`, `bank-additional-full.csv`, 41.188 rows
× 21 columns), decide per-client which contact **arm** to use, observe the reward (`y == "yes"`), and
update online — as opposed to a batch model whose decision is frozen until retrain.

Full task spec lives in `ref_doc/` (two PDFs — the official datathon brief and the team's own refined
plan). Read `ref_doc/Plano 7MLET Refinado.pdf` before making structural decisions; it defines the
target architecture, the data contract between stages, and the grading traps below.

Responsibilities per teammate (context for PR ownership, not enforced by tooling):
- **Gabriel**: repo setup, EDA, data pipeline (`src/data_prep.py`) — Etapas 0, 1, 2
- **Adryen**: bandit algorithms (`src/bandit.py`), baselines + replay simulation — Etapas 2, 3
- **Bertelli**: MLflow tracking, evaluation metrics, golden set, tests — Etapas 4, 7
- **Matheus**: FastAPI service (`src/api.py`), cloud architecture writeup — Etapas 5, 6

**Do not implement another teammate's Etapa without being asked.** `src/bandit.py` and `src/api.py`
are deliberately left as skeletons (docstring + `NotImplementedError`) — that is their correct state,
not an oversight.

## Current state

**Etapas 0, 1 and 2 are done.** Etapa 3 onward is not started.

```
data/raw/bank-additional-full.csv    # committed (41.188 × 21)
data/processed/bandit_frame.parquet  # Etapa 2 output (41.176 × 20) — committed
models/preprocessor.joblib           # fitted encoder, Etapa 2 output — committed
docs/data-dictionary.md              # raw + processed dictionary; read this before touching data
notebooks/01-eda.ipynb               # EDA, leakage analysis, arm choice (Etapa 1) — DONE
src/data_prep.py                     # Etapa 2 — DONE (the data contract)
src/tracking.py                      # MLflow setup — DONE
src/bandit.py                        # Etapa 3 — SKELETON ONLY (Adryen)
src/api.py                           # Etapa 5 — SKELETON ONLY (Matheus)
tests/                               # Etapa 4 — empty (Bertelli)
main.py                              # leftover from `uv init`, dead code — safe to delete
```

`src/data_prep.py` is load-bearing: the API, the bandit simulation and the golden-set evaluation must
all call the *same* cleaning/encoding functions instead of duplicating notebook logic (avoids
train/serving skew). Put pipeline code there, never inline in a notebook.

## Commands

Dependencies managed with `uv` (`pyproject.toml` + `uv.lock`, both committed).

```bash
uv sync                                   # install the venv from the lockfile
uv run jupyter lab                        # work in notebooks/
uv run python -m src.data_prep            # Etapa 2: raw → bandit_frame.parquet + preprocessor.joblib
uv run pytest -q                          # tests (tests/ still empty)
uv run mlflow-ui                          # MLflow UI → localhost:5000
uv add <package>                          # commit pyproject.toml + uv.lock together
```

Requires Python >= 3.13 (`.python-version`). `uv run uvicorn src.api:app --reload` will only work
once Etapa 5 exists.

## Environment constraints (learned the hard way — do not "fix" these)

- **`pandas` is pinned to `>=2.2,<3` on purpose.** Every MLflow 3.x requires `pandas<3`. When the
  project pinned `pandas>=3.0.3`, uv silently resolved `mlflow` down to **1.27.0** (a 2022 release),
  which is incompatible with protobuf 7 — `import mlflow` raised `ImportError` and `pytest` could not
  even collect. Bumping pandas to 3.x will silently break MLflow again.
- **MLflow uses a SQLite backend, not the `mlruns/` file store.** MLflow 3 deprecated the filesystem
  tracking backend and refuses it outright, so the plan's original `mlruns/` layout is obsolete. The
  DB and the artifacts both live under `mlruns/` (`mlruns/mlflow.db` + `mlruns/artifacts/`), gitignored.
- **Always configure MLflow through `src.tracking.setup_mlflow(experiment)`.** MLflow's default store
  is relative to the *cwd*, so a script run from the repo root and a notebook run from `notebooks/`
  would log to two different databases. `setup_mlflow` anchors the store at the repo root.
- **`[project.scripts]` needs the `[build-system]` block.** Without it uv treats the project as
  "virtual" and installs nothing — the `mlflow-ui` command would not exist.
- Notebook code must work on **pandas 2**: `select_dtypes(include=["str"])` is pandas-3-only, use
  `include=["object", "category"]`.

## Key domain facts (from EDA, `notebooks/01-eda.ipynb`)

- **Target is imbalanced**: ~11.3% `y == "yes"`. Never use accuracy; use conversion, PR-AUC, regret.
- **`duration` is leakage and must never be a model feature** — only known after the call ends
  (corr 0.40 with y). The plan calls this "desclassificação moral imediata" if it slips in. It is
  dropped in `clean()` and absent from `bandit_frame.parquet`. Keep asserting it.
- **`pdays == 999`** (96.3% of rows) is a sentinel for "never contacted before", not a number — it
  becomes the boolean `was_contacted_before` and the raw column is dropped.
- **`"unknown"` string values** are disguised missing data (default 20.9%, education 4.2%) — kept as
  their own category, never imputed (it's informative: "client declined to answer"). They survive
  one-hot encoding as their own columns (`cat__default_unknown`, …).
- **Macro columns** (`emp.var.rate`, `euribor3m`, `nr.employed`, `cons.price.idx`, `cons.conf.idx`)
  are collinear (0.91–0.97) and are a proxy for *when* the call happened, not for the client.
- **`month` has no year** — the same `may` spans 2008, 2009 and 2010. There is no date column: the
  only real temporal information is the **row order**.
- **Row order is chronological (May 2008 → Nov 2010) and must never be shuffled** — the replay
  simulation depends on it, and shuffling destroys the drift narrative.

### Bandit arms — settled decision, do not revisit

**Arms = `contact` (`cellular` vs `telephone`) → 2 arms.** Reward = `y == "yes"`.
Conversion: `cellular` **14.74%**, `telephone` **5.23%**. Baselines for Etapa 3: naive fixed rule
(~5.2%) and best historical arm / static oracle (~14.7%).

**`day_of_week` was tested as a cross (10 arms) and rejected** — see section 11.1 of the EDA. The
day effect is *real* inside `cellular` (χ²=24, p=0.0001) and noise inside `telephone` (p=0.37), but
it is **unresolvable**: separating the top arms would need ~120.000 samples per arm, and replay only
delivers ~412 (10 arms throw away ~80% of the usable sample). `day_of_week` is a **context feature**,
not an arm.

**`contact` is NOT a context feature.** It is the *action* the bandit chooses. Including it in
`CONTEXT_FEATURES` would hand the model the very decision it is supposed to recommend.

### Drift — treino and golden set are different economic regimes

The temporal split is not a random sample. Train (first 80%) vs golden set (last 20%):
`euribor3m` 4.27 → **1.03**, conversion **6.38% → 30.83%** (2008 crash, then rate cuts made term
deposits attractive). Two consequences: (a) it is the argument *for* the bandit — a batch model
trained on the first 80% would predict ~6% in a 31% world; (b) it is a **limitation to declare** —
any metric on the golden set looks great for the wrong reason (the period is easy, not the model good).

## Conventions to preserve

- **Never rebalance** (no SMOTE/over/undersampling). The bandit needs the *real* base rate (11.27%)
  and the *real* per-arm rates — resampling makes conversion, regret and replay fiction. Handle the
  imbalance with `class_weight="balanced"` and proper metrics, never by altering the distribution.
- **Never shuffle.** Splits are temporal (`temporal_split`), 80/20, no `stratify`.
- Log bandit params (priors, epsilon, seed) and metrics via MLflow — priors must be documented
  explicitly (`Beta(1,1)` = uninformative prior), it's a stated grading requirement.
- Report bandit results across multiple seeds (mean ± std), never a single run.
- If you write `import x`, declare `x` in `pyproject.toml` — don't rely on transitive deps.

## Git conventions

- **Never add `Co-Authored-By: Claude` to commit messages.**
- Work happens on `develop`; PRs go `develop` → `main`.
- `CLAUDE.md` is meant to live on `develop` and stay out of `main`.
