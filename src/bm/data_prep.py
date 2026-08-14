"""Pipeline de limpeza e preparação da base (Etapa 2).

Contrato de dados compartilhado por notebooks, simulação do bandit, golden set e API.
Toda transformação vive aqui: se a limpeza morar só no notebook, a API diverge do modelo
e o projeto ganha train/serving skew.

Decisões vindas da EDA (`notebooks/01-eda.ipynb`, Etapa 1):

- `duration` é vazamento (só é conhecida após a ligação) e **nunca** entra como feature.
- `pdays == 999` é sentinela de "nunca contatado", não número — vira flag booleana.
- `"unknown"` é ausente disfarçado e é mantido como **categoria própria** (é informativo).
- A ordem das linhas é cronológica (mai/2008 → nov/2010) e **não pode ser embaralhada**:
  a simulação de replay da Etapa 3 depende dela.
- Braços do bandit = `contact` (`cellular` vs `telephone`). Recompensa = `y == "yes"`.
- **Não rebalanceamos** (sem SMOTE/over/undersampling): o bandit precisa da taxa-base real
  (~11,3%) e das taxas reais por braço. Reamostrar tornaria o regret e o replay ficção.
  O desbalanceamento é tratado com `class_weight="balanced"` no modelo de propensão e com
  métricas adequadas (PR-AUC, regret), nunca alterando a distribuição dos dados.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EXPECTED_SHAPE: tuple[int, int] = (41188, 21)

ARM_COL: str = "contact"
ARMS: list[str] = ["cellular", "telephone"]
TARGET_COL: str = "y"
REWARD_COL: str = "reward"

LEAKY_COLS: list[str] = ["duration"]
PDAYS_SENTINEL: int = 999

# Features de contexto. `contact` fica de fora de propósito: ele é a AÇÃO que o bandit
# escolhe, não um atributo do cliente. Incluí-lo daria ao modelo de contexto a própria
# decisão que ele deveria recomendar.
CATEGORICAL_FEATURES: list[str] = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "month",
    "day_of_week",
    "poutcome",
]
NUMERIC_FEATURES: list[str] = [
    "age",
    "campaign",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]
BOOLEAN_FEATURES: list[str] = ["was_contacted_before"]

CONTEXT_FEATURES: list[str] = (
    CATEGORICAL_FEATURES + NUMERIC_FEATURES + BOOLEAN_FEATURES
)


def load_raw(path: str | Path) -> pd.DataFrame:
    """Lê o CSV bruto (separador `;`) e valida o shape esperado (41.188 × 21)."""
    df = pd.read_csv(path, sep=";")
    if df.shape != EXPECTED_SHAPE:
        raise ValueError(f"shape inesperado: {df.shape}, esperado {EXPECTED_SHAPE}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza canônica. Não reordena nem embaralha as linhas.

    - remove duplicatas exatas (12 na base original)
    - remove `duration` (vazamento)
    - cria `was_contacted_before` a partir de `pdays == 999` e descarta o `pdays` cru
    - mantém `"unknown"` como categoria própria (não imputa)
    """
    out = df.drop_duplicates().reset_index(drop=True)

    out["was_contacted_before"] = out["pdays"] != PDAYS_SENTINEL
    out = out.drop(columns=["pdays"])

    out = out.drop(columns=[c for c in LEAKY_COLS if c in out.columns])

    return out


def build_bandit_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Monta o frame consumido pela simulação de replay.

    Saída: `arm` (braço de fato usado no histórico), `reward` (0/1) e as features de contexto.
    A ordem cronológica das linhas é preservada — o replay depende dela.
    """
    if "duration" in df.columns:
        raise ValueError("`duration` e vazamento e nao pode chegar ao bandit_frame")

    frame = pd.DataFrame(index=df.index)
    frame["arm"] = df[ARM_COL]
    frame[REWARD_COL] = (df[TARGET_COL] == "yes").astype(int)

    for col in CONTEXT_FEATURES:
        frame[col] = df[col]

    return frame


def build_preprocessor() -> ColumnTransformer:
    """One-hot nas categóricas (+ `unknown` como categoria), padronização nas numéricas.

    `handle_unknown="ignore"` permite que a API receba uma categoria nunca vista sem quebrar.
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
        ],
        remainder="drop",
    )


def temporal_split(
    frame: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split cronológico, **sem shuffle**: primeiros (1-test_size) vs últimos test_size.

    Embaralhar quebraria a lógica online do replay e destruiria a narrativa de drift
    (a base cobre mai/2008 → nov/2010, atravessando a crise).
    """
    cut = int(len(frame) * (1 - test_size))
    return frame.iloc[:cut].copy(), frame.iloc[cut:].copy()


def arm_stats(frame: pd.DataFrame) -> pd.DataFrame:
    """Taxa de conversão observada por braço — baselines da Etapa 3."""
    return (
        frame.groupby("arm")[REWARD_COL]
        .agg(n="size", conversao="mean")
        .assign(conversao_pct=lambda d: (d.conversao * 100).round(2))
        .sort_values("conversao", ascending=False)
    )


def run_pipeline(
    raw_path: str | Path = "data/raw/bank-additional-full.csv",
    processed_dir: str | Path = "data/processed",
    models_dir: str | Path = "models",
    test_size: float = 0.2,
    experiment: str = "datathon-data-prep",
) -> pd.DataFrame:
    """Executa a Etapa 2 de ponta a ponta e registra tudo no MLflow.

    Persiste `bandit_frame.parquet` e `preprocessor.joblib` — os artefatos que a Etapa 3
    (simulação) e a Etapa 5 (API) consomem.
    """
    import mlflow

    from bm.tracking import setup_mlflow

    processed_dir, models_dir = Path(processed_dir), Path(models_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    setup_mlflow(experiment)
    with mlflow.start_run(run_name="data_prep"):
        raw = load_raw(raw_path)
        clean_df = clean(raw)
        frame = build_bandit_frame(clean_df)
        train, test = temporal_split(frame, test_size=test_size)

        preprocessor = build_preprocessor()
        preprocessor.fit(train[CONTEXT_FEATURES])

        frame_path = processed_dir / "bandit_frame.parquet"
        prep_path = models_dir / "preprocessor.joblib"
        frame.to_parquet(frame_path, index=False)
        joblib.dump(preprocessor, prep_path)

        stats = arm_stats(frame)
        n_features = preprocessor.transform(train[CONTEXT_FEATURES].head(1)).shape[1]

        mlflow.log_params(
            {
                "n_linhas_brutas": len(raw),
                "n_duplicatas_removidas": len(raw) - len(clean_df),
                "colunas_removidas_leakage": ",".join(LEAKY_COLS),
                "rebalanceamento": "nenhum (taxa-base real preservada)",
                "split": f"temporal sem shuffle, test_size={test_size}",
                "n_arms": frame.arm.nunique(),
                "arms": ",".join(sorted(frame.arm.unique())),
                "n_features_pos_encoding": n_features,
            }
        )
        mlflow.log_metrics(
            {
                "taxa_base_global": float(frame[REWARD_COL].mean()),
                "n_treino": len(train),
                "n_teste_golden": len(test),
                **{
                    f"conversao_{arm}": float(row.conversao)
                    for arm, row in stats.iterrows()
                },
            }
        )
        mlflow.log_artifact(str(frame_path), artifact_path="data")
        mlflow.log_artifact(str(prep_path), artifact_path="models")

    return frame


if __name__ == "__main__":
    frame = run_pipeline()
    print(f"bandit_frame: {frame.shape[0]} linhas x {frame.shape[1]} colunas")
    print(f"taxa-base global: {frame[REWARD_COL].mean() * 100:.2f}%\n")
    print(arm_stats(frame).to_string())
