# BankMarketing7mlet — Plataforma de Experimentação Adaptativa (Multi-Armed Bandit)

Datathon POSTECH/MLET — Grupo 61. O sistema **não é um classificador tradicional**: ele decide,
para cada cliente, **qual canal de contato usar**, observa a recompensa (o cliente converteu?) e
**atualiza sua crença online** — em vez de congelar a decisão até o próximo retreino.

| Pessoa | Frente | Etapas |
|---|---|---|
| Gabriel | Organização e Engenharia de Dados | 0, 1 e 2 |
| Adryen | Modelos (bandit) | 2 e 3 |
| Bertelli | Validação e MLOps | 4 e 7 |
| Matheus | Serviço e Infra | 5 e 6 |

### Progresso

| Etapa | Entregável | Status |
|---|---|---|
| 0 — Organização | estrutura, `uv`, README | ✅ concluída |
| 1 — EDA | `notebooks/01-eda.ipynb`, escolha dos braços | ✅ concluída |
| 2 — Preparação da base | `src/data_prep.py`, `bandit_frame.parquet`, `preprocessor.joblib` | ✅ concluída |
| 3 — Baseline e bandit | `src/bandit.py` | ⬜ esqueleto |
| 4 — Avaliação e golden set | `tests/` | ⬜ não iniciada |
| 5 — API | `src/api.py` | ⬜ esqueleto |
| 6 — Arquitetura em nuvem | seção 7 deste README | ⬜ não iniciada |
| 7 — MLOps / MLflow | instrumentação da Etapa 3 | ⬜ parcial (Etapa 2 já loga) |
| 8 — Apresentação | vídeo | ⬜ não iniciada |

---

## 1. O problema

<!-- responsável: Gabriel -->

Uma campanha de telemarketing bancário tem orçamento finito de ligações. A abordagem clássica —
regra fixa ("ligue sempre no telefone fixo") ou um teste A/B — desperdiça tráfego: o A/B mantém
metade dos clientes no braço ruim até o teste terminar, e a regra fixa nunca aprende.

Um **multi-armed bandit** equilibra **exploração** (testar braços incertos) e **explotação** (usar
o braço que parece melhor), realocando tráfego para o braço vencedor *enquanto* aprende.

**Formulação:**

- **Braços (ações):** `contact` → `cellular` ou `telephone` (**2 braços**).
- **Recompensa:** `y == "yes"` → 1 (cliente assinou o depósito a prazo), senão 0.
- **Avaliação offline:** *replay / rejection sampling* (Li et al., 2011) — percorremos o dataset
  na ordem cronológica; a cada linha o bandit escolhe um braço e **só contabilizamos a recompensa
  quando a escolha coincide com o braço de fato usado historicamente**. É a forma estatisticamente
  honesta de avaliar uma política de decisão com dados observacionais: só sabemos o desfecho da
  ação que realmente aconteceu.

## 2. Base de dados

<!-- responsável: Gabriel -->

[Bank Marketing (henriqueyamahata)](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)
— `bank-additional-full.csv`, **41.188 linhas × 21 colunas** (20 features + target `y`), separador `;`.
Campanhas reais de um banco português entre **maio/2008 e novembro/2010**.

> 📖 **Dicionário completo (raw + processed): [`docs/data-dictionary.md`](docs/data-dictionary.md).**
> Leia antes de mexer nos dados — é onde estão os domínios de cada coluna e as armadilhas.

Achados da EDA (`notebooks/01-eda.ipynb`) que **governam todo o pipeline**:

| Achado | Consequência |
|---|---|
| Target desbalanceado: **~11,3% de `yes`** | Acurácia é inútil. Usamos conversão, regret, PR-AUC. |
| **`duration` é vazamento** — só é conhecida *depois* da ligação (corr. 0,40 com `y`) | **Removida do conjunto de modelagem.** Usá-la é desclassificação. |
| **`pdays == 999`** em 96% das linhas é **sentinela**, não número | Convertida na flag booleana `was_contacted_before`. |
| **`"unknown"`** é ausente disfarçado (`default` 21%, `education` 4%) | Mantido como **categoria própria** — "o cliente não informou" é informativo. |
| Macro (`euribor3m`, `emp.var.rate`, `nr.employed`) colineares (0,91–0,97) | Proxy do período 2008–2010; sob vigilância por viés temporal. |
| **12 duplicatas exatas** | Removidas na limpeza. |
| Ordem das linhas é **cronológica** | **Nunca embaralhar** — a simulação de replay depende dela. |

**Escolha dos braços (com dados, não achismo):**

| Braço candidato | Conversão | Amplitude |
|---|---|---|
| `contact = cellular` | **14,74%** | **9,5 p.p.** ✅ escolhido |
| `contact = telephone` | **5,23%** | |
| `day_of_week` (mon–fri) | 9,95% – 12,11% | 2,2 p.p. ❌ descartado |

`cellular` converte quase **3x** mais que `telephone` — diferença grande o bastante para haver algo
real a descobrir. Isso define os dois baselines da Etapa 3: regra fixa ingênua (~5,2%) e melhor braço
histórico / oráculo estático (~14,7%). O bandit deve esmagar a primeira e convergir para perto do segundo.

**Por que não cruzamos com `day_of_week` (10 braços)?** Testamos a hipótese formalmente (seção 11.1 da
EDA) em vez de descartá-la por intuição. O sinal do dia é **real dentro do `cellular`** (χ²=24, p=0,0001:
segunda 12,8% vs terça 15,8%) e **ruído no `telephone`** (p=0,37) — mas o bandit não conseguiria
*resolver* os braços resultantes. Os três que disputam o topo (`cellular` em ter/qui/qua, 15,3%–15,8%)
têm intervalos de confiança sobrepostos: separá-los exigiria **~120.000 amostras por braço**, e o
dataset inteiro tem 41 mil linhas. Pior, o replay descarta as linhas em que o bandit diverge do
histórico, então 10 braços jogariam fora **~80% da amostra útil** (~20.600 → ~4.100 eventos).
O `day_of_week` **não é perdido**: entra como **feature de contexto** para a variante contextual da
Etapa 3, capturando o sinal sem fragmentar o espaço de ações.

## 3. Como executar

<!-- responsável: Matheus -->

Ambiente gerenciado com [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock` commitados).

```bash
uv sync                                   # cria o .venv a partir do lockfile
uv run jupyter lab                        # abre os notebooks
uv run python -m src.data_prep            # Etapa 2: raw -> bandit_frame.parquet + preprocessor.joblib
uv run pytest -q                          # testes
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # UI do MLflow (localhost:5000)
```

Requer Python >= 3.13. O CSV bruto já está em `data/raw/`; se faltar, o notebook de EDA o baixa do
Kaggle automaticamente (requer `~/.kaggle/kaggle.json`).

> **Estado atual: Etapas 0, 1 e 2 concluídas.** `src/bandit.py` (Etapa 3) e `src/api.py` (Etapa 5)
> ainda são esqueletos — o comando `uv run uvicorn src.api:app --reload` só funcionará depois da
> Etapa 5.

<!-- TODO Matheus: exemplos de curl para /recommend e /feedback + screenshot do Swagger -->

## 4. Formulação do bandit

<!-- responsável: Adryen -->
<!-- TODO: Epsilon-Greedy vs Thompson Sampling; priors Beta(1,1) = uniforme/não-informativo -->

## 5. Resultados

<!-- responsável: Adryen -->
<!-- TODO: tabela conversão/regret por política, gráfico de conversão acumulada,
     análise exploração × explotação, média ± desvio sobre múltiplas seeds -->

## 6. Golden Set

<!-- responsável: Bertelli -->
<!-- TODO: 5 clientes de teste, recomendação do sistema e justificativa de cada um -->

## 7. Arquitetura em nuvem

<!-- responsável: Matheus -->
<!-- TODO: mapeamento 1:1 dos artefatos do repo para serviços gerenciados -->

## 8. MLOps (ciclo de vida)

<!-- responsável: Bertelli -->

O MLflow é configurado **sempre** via `src.tracking.setup_mlflow(experimento)`. Duas armadilhas que
isso resolve:

- O store padrão do MLflow é **relativo ao diretório de execução** — um script rodado da raiz e um
  notebook rodado de `notebooks/` gravariam em bancos diferentes. `setup_mlflow` ancora o store na
  raiz do repositório.
- O **MLflow 3 descontinuou o file store `mlruns/`** (que o plano original previa) e recusa o
  backend de filesystem. Usamos SQLite (`mlflow.db`); `mlruns/` guarda só os artefatos.

A Etapa 2 já loga (parâmetros da limpeza, taxa-base, conversão por braço, `preprocessor.joblib` e
`bandit_frame.parquet` como artefatos). Ver a UI com:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

<!-- TODO Bertelli: instrumentar a Etapa 3 (priors, epsilon, seed, regret, n_matched) e escrever o
     parágrafo do ciclo: dados → experimento → estado do bandit → serving → feedback →
     monitoramento → reset em drift -->

## 9. Limitações

<!-- responsável: todos -->

**O golden set não é uma amostra representativa da base.** O split é temporal, e treino e golden set
caem em regimes econômicos diferentes:

| | Treino (80% inicial) | Golden set (20% final) |
|---|---|---|
| `euribor3m` | 4,27 | **1,03** |
| `emp.var.rate` | 0,65 | **−2,19** |
| **conversão** | **6,38%** | **30,83%** |

Com o crash de 2008 e o corte de juros que veio depois, o depósito a prazo perdeu concorrência e a
conversão quase **quintuplicou**. Isso corta dos dois lados. É o **argumento a favor do bandit**: um
classificador batch treinado nos primeiros 80% aprenderia "conversão ≈ 6%" e chegaria em 2010
prevendo 6% num mundo de 31% — completamente descalibrado, enquanto o bandit se readapta a cada
feedback. Mas é também um **alerta de leitura**: qualquer métrica medida no golden set vai parecer
excelente pelo motivo errado — o período é fácil, não o modelo é bom.

<!-- TODO: avaliação offline por replay ≠ produção (só sabemos o desfecho da ação que de fato
     aconteceu); dados de 2008–2010 não representam o mercado atual -->

---

## Estrutura do repositório

```
data/raw/                       # bank-additional-full.csv (commitado)
data/processed/
  bandit_frame.parquet          # saída da Etapa 2: arm + reward + 18 features de contexto
docs/data-dictionary.md         # dicionário raw + processed
notebooks/
  01-eda.ipynb                  # EDA, leakage e escolha dos braços (Etapa 1)          ✅
src/
  data_prep.py                  # contrato de dados: load_raw/clean/build_bandit_frame  ✅
  tracking.py                   # configuração única do MLflow                          ✅
  bandit.py                     # EpsilonGreedy / ThompsonSampling (Etapa 3)     ⬜ esqueleto
  api.py                        # FastAPI: /recommend, /feedback, /health (Etapa 5) ⬜ esqueleto
models/
  preprocessor.joblib           # encoder ajustado no treino (Etapa 2)
  bandit_state.json             # estado do bandit (Etapa 5) — ainda não existe
reports/figures/                # gráficos usados no README e no vídeo
tests/                          # pytest (Etapa 4) — ainda vazio
mlflow.db / mlruns/             # MLflow: banco + artefatos (não versionados)
```

`src/` existe justamente para que a API, a simulação do bandit e a avaliação chamem **as mesmas**
funções de limpeza/encoding — em vez de duplicar a lógica do notebook e criar train/serving skew.
