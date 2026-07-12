"""Configuração única do MLflow (Etapa 7).

Por que este módulo existe — dois problemas reais, resolvidos num lugar só:

1. **O store é relativo ao cwd.** O MLflow usa como padrão um caminho *relativo ao diretório de
   onde o processo foi executado*. Rodar um script da raiz e um notebook de `notebooks/` gravaria
   em dois bancos diferentes, e os runs do grupo se espalhariam sem ninguém perceber. Aqui o store
   é ancorado na **raiz do repositório**: notebook, script, API e testes registram sempre no mesmo
   lugar, independente do cwd.

2. **O file store (`mlruns/`) foi descontinuado no MLflow 3.** O plano original previa `mlruns/`
   como backend, mas o MLflow 3 recusa o filesystem backend ("maintenance mode") e exige um banco.
   Usamos SQLite — local, sem servidor. Banco e artefatos ficam ambos sob `mlruns/`.

Uso no código:

    from src.tracking import setup_mlflow
    setup_mlflow("datathon-bandit")

Abrir a UI (de qualquer diretório):

    uv run mlflow-ui        # → http://localhost:5000
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

MLRUNS_DIR: Path = REPO_ROOT / "mlruns"
DB_PATH: Path = MLRUNS_DIR / "mlflow.db"
ARTIFACTS_DIR: Path = MLRUNS_DIR / "artifacts"

TRACKING_URI: str = f"sqlite:///{DB_PATH}"


def setup_mlflow(experiment: str) -> str:
    """Ancora o MLflow na raiz do repo e seleciona o experimento. Devolve o tracking URI."""
    import mlflow

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)

    # O artifact_location só pode ser fixado na criação do experimento — depois disso é imutável.
    # Sem isso os artefatos cairiam num diretório relativo ao cwd de quem rodou.
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=ARTIFACTS_DIR.as_uri())
    mlflow.set_experiment(experiment)
    return TRACKING_URI


def launch_ui() -> int:
    """Sobe a UI do MLflow apontando para o store do projeto. Entry point: `uv run mlflow-ui`."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    port = sys.argv[1] if len(sys.argv) > 1 else "5000"

    print(f"MLflow UI  ->  http://localhost:{port}", flush=True)
    print(f"store      ->  {DB_PATH}", flush=True)
    print("(a UI leva ~10s para subir; ctrl+c para parar)\n", flush=True)

    try:
        return subprocess.call(
            [
                "mlflow",
                "ui",
                "--backend-store-uri",
                TRACKING_URI,
                "--default-artifact-root",
                ARTIFACTS_DIR.as_uri(),
                "--port",
                port,
            ]
        )
    except KeyboardInterrupt:
        return 0
