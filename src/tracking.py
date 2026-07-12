"""Configuração única do MLflow (Etapa 7).

Por que este módulo existe — dois problemas reais, resolvidos num lugar só:

1. **O store é relativo ao cwd.** O MLflow usa como padrão `sqlite:///mlflow.db` *relativo ao
   diretório de onde o processo foi executado*. Rodar um script da raiz e um notebook de
   `notebooks/` gravaria em dois bancos diferentes, e os runs do grupo se espalhariam. Aqui o
   store é ancorado na **raiz do repositório**: notebook, script, API e testes registram sempre
   no mesmo lugar, independente do cwd.

2. **O file store (`mlruns/`) foi descontinuado no MLflow 3.** O plano original previa `mlruns/`
   como backend, mas o MLflow 3 recusa o filesystem backend ("maintenance mode") e exige um
   banco. Usamos SQLite (`mlflow.db`), que é local, não precisa de servidor e ainda habilita o
   model registry. Os **artefatos** continuam em `mlruns/`.

Uso:

    from src.tracking import setup_mlflow
    setup_mlflow("datathon-bandit")

Visualizar a UI (de qualquer diretório):

    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # → http://localhost:5000
"""

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DB_PATH: Path = REPO_ROOT / "mlflow.db"
ARTIFACTS_DIR: Path = REPO_ROOT / "mlruns"

TRACKING_URI: str = f"sqlite:///{DB_PATH}"


def setup_mlflow(experiment: str) -> str:
    """Ancora o MLflow na raiz do repo e seleciona o experimento. Devolve o tracking URI."""
    import mlflow

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)

    # O artifact_location só pode ser fixado na criação do experimento — depois disso ele é
    # imutável. Sem isso os artefatos cairiam num `mlruns/` relativo ao cwd de quem rodou.
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=ARTIFACTS_DIR.as_uri())
    mlflow.set_experiment(experiment)
    return TRACKING_URI
