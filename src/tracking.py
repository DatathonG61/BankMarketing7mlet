"""Configuração única do MLflow (Etapa 7).

Um único ponto de configuração para todo o projeto — notebook, script, API e testes. Chame
sempre `setup_mlflow(...)`, nunca `mlflow.set_tracking_uri()` direto.

## Dois modos

**Compartilhado (o do grupo).** Se existir `MLFLOW_TRACKING_URI` no ambiente (via `.env`), os
runs vão para o servidor MLflow do VPS e os quatro enxergam os mesmos experimentos. Copie o
`.env.example` para `.env` e preencha — o `.env` **não** é versionado (tem senha).

**Local (fallback).** Sem `.env`, tudo cai num SQLite em `mlruns/mlflow.db`. Serve para rodar
offline sem depender do servidor. Os runs ficam só na sua máquina.

## Por que não commitar o mlflow.db

SQLite é binário: o git não sabe mesclar. Dois colegas rodando experimentos no mesmo dia
produziriam um merge que sobrescreve os runs de um dos dois. E SQLite em pasta sincronizada
(Google Drive/Dropbox) **corrompe** — o sync não respeita as travas de arquivo. Por isso o
compartilhamento é feito por um servidor, não pelo repositório.

## Detalhe que este módulo resolve

O store padrão do MLflow é relativo ao **cwd**. Sem ancorar, um script rodado da raiz e um
notebook rodado de `notebooks/` gravariam em bancos diferentes, e os runs se espalhariam sem
ninguém perceber. Aqui o caminho local é sempre absoluto, a partir da raiz do repo.

## Uso

    from src.tracking import setup_mlflow
    setup_mlflow("datathon-bandit")

    uv run mlflow-ui        # abre a UI (local ou remota, conforme o .env)
"""

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

MLRUNS_DIR: Path = REPO_ROOT / "mlruns"
DB_PATH: Path = MLRUNS_DIR / "mlflow.db"
ARTIFACTS_DIR: Path = MLRUNS_DIR / "artifacts"

LOCAL_TRACKING_URI: str = f"sqlite:///{DB_PATH}"


def _load_dotenv() -> None:
    """Carrega o `.env` da raiz do repo, se existir. Não sobrescreve variáveis já definidas."""
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def tracking_uri() -> str:
    """URI do servidor compartilhado (se houver `.env`), senão o SQLite local."""
    _load_dotenv()
    return os.environ.get("MLFLOW_TRACKING_URI") or LOCAL_TRACKING_URI


def is_remote() -> bool:
    """True se estamos apontando para o servidor do grupo."""
    return tracking_uri().startswith(("http://", "https://"))


def setup_mlflow(experiment: str) -> str:
    """Configura o MLflow e seleciona o experimento. Devolve o tracking URI em uso."""
    import mlflow

    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)

    if is_remote():
        # No servidor, o artifact root é definido por ele (--serve-artifacts).
        mlflow.set_experiment(experiment)
        return uri

    # Local: o artifact_location só pode ser fixado na criação do experimento — depois é imutável.
    # Sem isso os artefatos cairiam num diretório relativo ao cwd de quem rodou.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=ARTIFACTS_DIR.as_uri())
    mlflow.set_experiment(experiment)
    return uri


def launch_ui() -> int:
    """Abre a UI do MLflow. Entry point: `uv run mlflow-ui`.

    Com `.env` configurado, apenas abre o navegador no servidor do grupo (a UI já roda lá).
    Sem `.env`, sobe um servidor local contra `mlruns/mlflow.db`.
    """
    uri = tracking_uri()

    if is_remote():
        print(f"MLflow do grupo  ->  {uri}")
        print("(a UI ja roda no servidor; abrindo no navegador)")
        webbrowser.open(uri)
        return 0

    port = sys.argv[1] if len(sys.argv) > 1 else "5000"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("MLflow LOCAL (sem .env — os runs ficam so nesta maquina)", flush=True)
    print(f"  UI     ->  http://localhost:{port}", flush=True)
    print(f"  store  ->  {DB_PATH}", flush=True)
    print("  (leva ~10s para subir; ctrl+c para parar)\n", flush=True)

    try:
        return subprocess.call(
            [
                "mlflow",
                "ui",
                "--backend-store-uri",
                LOCAL_TRACKING_URI,
                "--default-artifact-root",
                ARTIFACTS_DIR.as_uri(),
                "--port",
                port,
            ]
        )
    except KeyboardInterrupt:
        return 0
