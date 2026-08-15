"""Análise de exploração x explotação e gráficos finais — Etapa 3.4.

Entregável cobrado **em texto** pelo enunciado, na tabela de referências algorítmicas
(coluna "evidência esperada"): "análise de exploração" para o Thompson Sampling e
"análise do trade-off entre exploração e conversão" para o Epsilon-Greedy. As métricas
numéricas da Etapa 4.1 já respondiam "qual política converte mais"; o que faltava era
mostrar *como* cada uma chega lá — quanto tráfego ela gasta testando o braço pior antes
de convergir.

Gera quatro figuras em `reports/figures/`:

1. `conversao_acumulada.png` — conversão acumulada de cada política contra os dois
   baselines (regra fixa e oráculo estático).
2. `escolha_por_braco.png` — fração de escolha de cada braço ao longo das rodadas. É o
   gráfico que conta a história da exploração: começa perto de 50/50 e converge.
3. `regret_acumulado.png` — regret acumulado por política. Curva que achata = a política
   parou de pagar pelo aprendizado.
4. `posterior_thompson.png` — evolução da crença do Thompson sobre cada braço, medida no
   início, meio e fim do replay.

**Sobre a figura 4:** o plano do grupo previa plotar as densidades Beta de um Thompson
Beta-Bernoulli. O Thompson que o grupo implementou é *contextual* (regressão linear
bayesiana por braço), então não existe uma Beta para plotar — a crença é uma normal
multivariada sobre coeficientes, não uma distribuição sobre uma taxa escalar. O
equivalente honesto, e o que está plotado aqui, é a distribuição do score posterior
médio (`mu_braco @ x`) sobre uma amostra fixa de clientes: se a crença está aprendendo,
as duas distribuições se separam ao longo do tempo. Conta a mesma história que a Beta
contaria.

**Seeds:** as figuras usam menos seeds que a tabela da Etapa 4.1 (que roda 10). Um replay
completo sobre as 41 mil linhas custa alguns minutos por seed no Thompson contextual, e
para o formato "média + faixa de desvio" das curvas 3 seeds já mostram a dispersão. Os
*números* citados no README continuam vindo das 10 seeds de `run_thompson_replay` /
`run_epsilon_replay` — estas figuras ilustram o comportamento, não substituem a tabela.

Uso:

    uv run python -m bm.exploration              # 3 seeds, salva as 4 figuras
    uv run python -m bm.exploration --seeds 5    # mais seeds, faixa mais confiável
    uv run python -m bm.exploration --no-mlflow  # não registra as figuras no MLflow
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# Backend sem janela: este módulo roda em terminal e em CI, nunca abre display.
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bm.data_prep import ARMS, CONTEXT_FEATURES, arm_stats
from bm.evaluation import compute_regret
from bm.experiments.train import train_bandit
from bm.models.bandit import EpsilonGreedy, ThompsonSampling

FIGURES_DIR = Path("reports/figures")

ALPHA_PRIOR: float = 1.0
EPSILON: float = 0.1
DEFAULT_SEEDS: int = 3
ROLLING_WINDOW: int = 2000

# Paleta fixa por braço: a mesma cor identifica `cellular` nas quatro figuras.
CORES_BRACO = {"cellular": "#1F6FB4", "telephone": "#C4622D"}
CORES_POLITICA = {"Thompson Sampling": "#3D3A9E", "Epsilon-Greedy": "#C4622D"}


def _dense(matrix) -> np.ndarray:
    """Preprocessor pode devolver matriz esparsa dependendo da versão do sklearn."""
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _n_features(frame: pd.DataFrame, preprocessor) -> int:
    return _dense(preprocessor.transform(frame[CONTEXT_FEATURES].head(1))).shape[1]


def _make_bandit(politica: str, seed: int, n_features: int):
    if politica == "Thompson Sampling":
        return ThompsonSampling(alpha=ALPHA_PRIOR, n_features=n_features, seed=seed)
    return EpsilonGreedy(epsilon=EPSILON, seed=seed)


def collect_traces(
    frame: pd.DataFrame, preprocessor, politica: str, seeds: list[int]
) -> list[list[tuple]]:
    """Roda o replay uma vez por seed e devolve o trace completo de cada rodada.

    Usa o mesmo `train_bandit` da Etapa 3 que a tabela de métricas usa — as curvas não
    podem sair de uma reimplementação paralela do replay, senão figura e tabela contam
    histórias diferentes sobre a mesma simulação.
    """
    n_features = _n_features(frame, preprocessor)
    traces = []
    for seed in seeds:
        bandit = _make_bandit(politica, seed, n_features)
        trace: list[tuple] = []
        train_bandit(bandit, frame, preprocessor, log_to_mlflow=False, trace=trace)
        traces.append(trace)
        n_matched = sum(t[2] for t in trace)
        print(f"  {politica} seed={seed}: {n_matched} linhas aproveitadas de {len(trace)}")
    return traces


def _matched_rewards(trace: list[tuple]) -> np.ndarray:
    return np.array([reward for _, _, matched, reward in trace if matched], dtype=float)


def _matched_steps(trace: list[tuple]) -> np.ndarray:
    """Posição no dataset de cada rodada que sobreviveu ao rejection sampling."""
    return np.array([step for step, _, matched, _ in trace if matched], dtype=float)


# Rodadas iniciais descartadas na plotagem das curvas acumuladas: com 10 ou 20 amostras
# a média acumulada oscila entre 0% e 100% e domina a escala do gráfico sem dizer nada.
BURN_IN: int = 100


def _curva_no_grid(trace: list[tuple], valores: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Reamostra uma série indexada por rodada aproveitada para o eixo de steps do dataset.

    Sem isso as políticas não são comparáveis: o Thompson aproveita ~21 mil linhas e o
    Epsilon-Greedy ~36 mil, então a "rodada 10.000" de cada um cai num ponto diferente da
    campanha. Ancorar tudo na posição real dentro da base coloca as duas curvas no mesmo
    tempo cronológico -- e é o único jeito de a comparação com os baselines significar
    alguma coisa.
    """
    steps = _matched_steps(trace)[BURN_IN:]
    return np.interp(grid, steps, valores[BURN_IN:])


def _grid_de_steps(n_linhas: int, pontos: int = 400) -> np.ndarray:
    return np.linspace(n_linhas * 0.05, n_linhas, pontos)


def _stack(series: list[np.ndarray]) -> np.ndarray:
    return np.vstack(series)


def _stack_truncated(series: list[np.ndarray]) -> np.ndarray:
    """Empilha séries de comprimentos diferentes cortando todas na mais curta.

    Usado só onde as séries já compartilham o mesmo eixo (a janela móvel percorre todas
    as rodadas, não só as aproveitadas, então todas as seeds têm o mesmo comprimento a
    menos de arredondamento).
    """
    menor = min(len(s) for s in series)
    return np.vstack([s[:menor] for s in series])


def _banda(ax, x, curvas: np.ndarray, cor: str, rotulo: str) -> None:
    """Linha da média sobre as seeds + faixa de mais ou menos um desvio-padrão."""
    media = curvas.mean(axis=0)
    desvio = curvas.std(axis=0)
    ax.plot(x, media, color=cor, linewidth=1.8, label=rotulo, zorder=3)
    ax.fill_between(x, media - desvio, media + desvio, color=cor, alpha=0.16, linewidth=0)


def _estilo(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#CCCCCC", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)


def plot_conversao_acumulada(
    traces_por_politica: dict[str, list[list[tuple]]],
    stats: pd.DataFrame,
    n_linhas: int,
    destino: Path,
) -> Path:
    """Conversão acumulada de cada política contra os dois baselines.

    As duas linhas horizontais são médias do **período inteiro**; as curvas são
    acumuladas *até* cada ponto do tempo. Como a conversão da base sobe muito no fim
    (2,9% no começo, mais de 17% no fim), as curvas cruzam a linha da regra fixa por
    efeito do calendário, não só por mérito da política. A comparação honesta é entre as
    duas curvas, que compartilham o mesmo eixo de tempo.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    p_fixa = stats.loc["telephone", "conversao"]
    p_oraculo = stats.loc["cellular", "conversao"]
    grid = _grid_de_steps(n_linhas)

    for politica, traces in traces_por_politica.items():
        curvas = _stack(
            [
                _curva_no_grid(
                    t,
                    np.cumsum(_matched_rewards(t))
                    / np.arange(1, len(_matched_rewards(t)) + 1),
                    grid,
                )
                for t in traces
            ]
        )
        _banda(ax, grid, curvas, CORES_POLITICA[politica], politica)

    ax.axhline(p_oraculo, color="#1F7A4D", linestyle="--", linewidth=1.3,
               label=f"Oráculo estático — sempre cellular ({p_oraculo * 100:.2f}% no período todo)")
    ax.axhline(p_fixa, color="#B3261E", linestyle=":", linewidth=1.3,
               label=f"Regra fixa — sempre telephone ({p_fixa * 100:.2f}% no período todo)")

    ax.set_xlabel("Posição na base (ordem cronológica, mai/2008 → nov/2010)")
    ax.set_ylabel("Conversão acumulada")
    ax.set_title("Conversão acumulada: bandits contra os baselines\n"
                 "média sobre as seeds, faixa = ± 1 desvio-padrão", loc="left")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _estilo(ax)

    fig.tight_layout()
    fig.savefig(destino, dpi=160)
    plt.close(fig)
    return destino


def plot_escolha_por_braco(
    traces_por_politica: dict[str, list[list[tuple]]], destino: Path
) -> Path:
    """Fração de escolha de cada braço ao longo das rodadas — o gráfico da exploração.

    Conta **todas** as rodadas, não só as aproveitadas: o que interessa aqui é a decisão
    da política, e ela decide em toda linha, mesmo quando o replay depois descarta.
    """
    politicas = list(traces_por_politica)
    fig, axes = plt.subplots(1, len(politicas), figsize=(11, 4.4), sharey=True)
    if len(politicas) == 1:
        axes = [axes]

    for ax, politica in zip(axes, politicas):
        traces = traces_por_politica[politica]
        for braco in ARMS:
            curvas = _stack_truncated(
                [
                    pd.Series([escolhido == braco for _, escolhido, _, _ in t])
                    .rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW)
                    .mean()
                    .dropna()
                    .to_numpy()
                    for t in traces
                ]
            )
            x = np.arange(ROLLING_WINDOW, ROLLING_WINDOW + curvas.shape[1])
            _banda(ax, x, curvas, CORES_BRACO[braco], braco)

        ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1,
                   label="50% = indecisão total")
        ax.set_title(politica, loc="left", fontsize=11)
        ax.set_xlabel("Rodada do replay")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
        _estilo(ax)

    axes[0].set_ylabel(f"Fração de escolha (janela móvel de {ROLLING_WINDOW})")
    axes[-1].legend(frameon=False, fontsize=9, loc="center right")
    fig.suptitle("Exploração x explotação: como cada política distribui o tráfego",
                 x=0.008, ha="left", fontsize=12)

    fig.tight_layout()
    fig.savefig(destino, dpi=160)
    plt.close(fig)
    return destino


def plot_regret_acumulado(
    traces_por_politica: dict[str, list[list[tuple]]],
    p_oracle: float,
    n_linhas: int,
    destino: Path,
) -> Path:
    """Regret acumulado por política — curva que achata significa que aprendeu."""
    fig, ax = plt.subplots(figsize=(9, 5))
    grid = _grid_de_steps(n_linhas)

    for politica, traces in traces_por_politica.items():
        curvas = _stack(
            [
                _curva_no_grid(
                    t, np.asarray(compute_regret(_matched_rewards(t), p_oracle)), grid
                )
                for t in traces
            ]
        )
        _banda(ax, grid, curvas, CORES_POLITICA[politica], politica)

    ax.set_xlabel("Posição na base (ordem cronológica, mai/2008 → nov/2010)")
    ax.set_ylabel("Regret acumulado")
    ax.set_title("Regret acumulado contra o oráculo estático\n"
                 "a queda no trecho final é drift: a conversão do período supera a média "
                 "do oráculo (14,74%)", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _estilo(ax)

    fig.tight_layout()
    fig.savefig(destino, dpi=160)
    plt.close(fig)
    return destino


def posterior_snapshots(
    frame: pd.DataFrame,
    preprocessor,
    seed: int,
    n_checkpoints: int = 3,
    amostra: int = 3000,
) -> list[dict[str, np.ndarray]]:
    """Crença do Thompson sobre cada braço no início, meio e fim do replay.

    Treina em segmentos consecutivos do mesmo frame reaproveitando a mesma instância do
    bandit — o estado é cumulativo, então treinar em três fatias seguidas é idêntico a
    treinar de uma vez, só que com pontos de parada para fotografar a crença.
    """
    n_features = _n_features(frame, preprocessor)
    bandit = ThompsonSampling(alpha=ALPHA_PRIOR, n_features=n_features, seed=seed)

    x_amostra = _dense(
        preprocessor.transform(
            frame[CONTEXT_FEATURES].sample(min(amostra, len(frame)), random_state=seed)
        )
    )

    limites = np.linspace(0, len(frame), n_checkpoints + 1).astype(int)
    snapshots = []
    for i in range(n_checkpoints):
        segmento = frame.iloc[limites[i]: limites[i + 1]]
        train_bandit(bandit, segmento, preprocessor, log_to_mlflow=False)
        snapshots.append(
            {
                arm: x_amostra @ (estado.a_inv @ estado.b)
                for arm, estado in bandit.arms.items()
            }
        )
    return snapshots


def plot_posterior_thompson(
    snapshots: list[dict[str, np.ndarray]], destino: Path
) -> Path:
    """Distribuição do score posterior por braço em cada checkpoint."""
    rotulos = ["Início do replay", "Meio do replay", "Fim do replay"]
    fig, axes = plt.subplots(1, len(snapshots), figsize=(11, 3.9), sharex=True, sharey=True)
    if len(snapshots) == 1:
        axes = [axes]

    for i, (ax, scores) in enumerate(zip(axes, snapshots)):
        for braco in ARMS:
            ax.hist(scores[braco], bins=60, color=CORES_BRACO[braco], alpha=0.55,
                    label=braco, density=True)
        for braco in ARMS:
            ax.axvline(float(np.mean(scores[braco])), color=CORES_BRACO[braco],
                       linewidth=1.6, linestyle="--")
        titulo = rotulos[i] if i < len(rotulos) else f"Checkpoint {i + 1}"
        ax.set_title(titulo, loc="left", fontsize=11)
        ax.set_xlabel("Score posterior médio do braço")
        _estilo(ax)

    axes[0].set_ylabel("Densidade")
    axes[-1].legend(frameon=False, fontsize=9)
    fig.suptitle("Thompson contextual: como a crença sobre cada braço evolui no replay "
                 "(tracejado = média)", x=0.008, ha="left", fontsize=12)

    fig.tight_layout()
    fig.savefig(destino, dpi=160)
    plt.close(fig)
    return destino


def resumo_exploracao(traces_por_politica: dict[str, list[list[tuple]]]) -> pd.DataFrame:
    """Tabela numérica do trade-off — o texto que acompanha as figuras no README.

    "Tráfego no braço pior" é a leitura de negócio da exploração: a fração de ligações
    que a política gastou em `telephone`, o braço de conversão mais baixa.
    """
    linhas = []
    for politica, traces in traces_por_politica.items():
        gasto_pior = [
            float(np.mean([escolhido == "telephone" for _, escolhido, _, _ in t]))
            for t in traces
        ]
        conversao = [float(_matched_rewards(t).mean()) for t in traces]
        # Últimos 20% das rodadas: quanto a política ainda explora depois de aprender.
        gasto_final = [
            float(np.mean([escolhido == "telephone"
                           for _, escolhido, _, _ in t[int(len(t) * 0.8):]]))
            for t in traces
        ]
        linhas.append(
            {
                "Política": politica,
                "Tráfego no braço pior (total)": f"{np.mean(gasto_pior) * 100:.1f}%",
                "Tráfego no braço pior (últimos 20%)": f"{np.mean(gasto_final) * 100:.1f}%",
                "Conversão no replay": f"{np.mean(conversao) * 100:.2f}%",
            }
        )
    return pd.DataFrame(linhas)


def main(n_seeds: int = DEFAULT_SEEDS, usar_mlflow: bool = True) -> None:
    frame = pd.read_parquet("data/processed/bandit_frame.parquet")
    preprocessor = joblib.load("models/preprocessor.joblib")

    stats = arm_stats(frame)
    p_oracle = stats["conversao"].max()
    seeds = list(range(n_seeds))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    traces_por_politica = {}
    for politica in ("Thompson Sampling", "Epsilon-Greedy"):
        print(f"replay de {politica} em {n_seeds} seed(s)...")
        traces_por_politica[politica] = collect_traces(frame, preprocessor, politica, seeds)

    figuras = [
        plot_conversao_acumulada(traces_por_politica, stats, len(frame),
                                 FIGURES_DIR / "conversao_acumulada.png"),
        plot_escolha_por_braco(traces_por_politica,
                               FIGURES_DIR / "escolha_por_braco.png"),
        plot_regret_acumulado(traces_por_politica, p_oracle, len(frame),
                              FIGURES_DIR / "regret_acumulado.png"),
    ]

    print("checkpoints da crença do Thompson...")
    figuras.append(
        plot_posterior_thompson(
            posterior_snapshots(frame, preprocessor, seed=0),
            FIGURES_DIR / "posterior_thompson.png",
        )
    )

    tabela = resumo_exploracao(traces_por_politica)
    print()
    print(tabela.to_string(index=False))
    print()
    for figura in figuras:
        print(f"figura salva em {figura}")

    if usar_mlflow:
        import mlflow

        from bm.tracking import setup_mlflow

        setup_mlflow("datathon-bandit")
        with mlflow.start_run(run_name="analise_exploracao_etapa_3_4"):
            mlflow.log_params({"n_seeds": n_seeds, "rolling_window": ROLLING_WINDOW,
                               "epsilon": EPSILON, "alpha_prior": ALPHA_PRIOR})
            for figura in figuras:
                mlflow.log_artifact(str(figura), artifact_path="figures")
        print("figuras registradas no MLflow (experimento datathon-bandit)")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Análise de exploração da Etapa 3.4")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help=f"número de seeds por política (padrão: {DEFAULT_SEEDS})")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="não registrar as figuras no MLflow")
    args = parser.parse_args()

    main(n_seeds=args.seeds, usar_mlflow=not args.no_mlflow)
