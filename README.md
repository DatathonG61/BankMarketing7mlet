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
| 3 — Baseline e bandit | `src/bm/models/bandit.py`, `src/bm/experiments/` | 🔶 parcial (Thompson pronto e treinado; Epsilon-Greedy ainda esqueleto) |
| 4 — Avaliação e golden set | `tests/`, `src/bm/evaluation.py`, `src/bm/golden_set.py` | 🔶 parcial (tabela e golden set com Thompson; falta a linha de Epsilon-Greedy) |
| 5 — API | `src/api.py` | ⬜ esqueleto |
| 6 — Arquitetura em nuvem | seção 7 deste README | ⬜ não iniciada |
| 7 — MLOps / MLflow | `src/bm/mlflow_logging.py`, `src/bm/experiments/run_thompson_replay.py` | ✅ Thompson instrumentado (10 seeds, média ± desvio) |
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
uv run mlflow-ui                          # abre a UI do MLflow (localhost:5000)
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

### Tabela comparativa (Etapa 4.1 — Bertelli)

As duas primeiras linhas vêm direto do histórico (`arm_stats`, sem simular nada — é a taxa de
conversão real de quem sempre usou aquele canal). A linha do Thompson Sampling é a média ± desvio
de 10 seeds, via replay/rejection sampling sobre o `bandit_frame` completo (`run_thompson_replay`).
Epsilon-Greedy fica pendente até o Adryen implementar a classe (Etapa 3).

| Política | Conversão (replay) | Regret acumulado | N amostras usadas |
|---|---|---|---|
| Regra fixa (telephone) | 5,23% | alto | 15.041 |
| Melhor braço histórico (cellular) | 14,74% | 0 (oráculo) | 26.135 |
| Epsilon-Greedy (epsilon=0.1) | pendente | pendente | pendente |
| Thompson Sampling (média de 10 seeds) | 12,68% ± 0,23pp | 446,9 ± 46,3 | 21.714 |

O Thompson esmaga a regra fixa (5,23% → ~12,7%) e chega perto do oráculo (14,74%) sem nunca ter
recebido a informação de qual braço é o melhor — descobre sozinho, pagando o preço da exploração
(por isso não chega nos 14,74%: parte das rodadas ele ainda testa `telephone`). `N amostras usadas`
é menor que os baselines porque o replay descarta toda linha em que o braço escolhido pelo bandit
não bate com o braço real do histórico (rejection sampling — não dá pra saber o resultado de uma
ação que não foi de fato tomada).

Gerar/atualizar esta tabela: `uv run python -m bm.evaluation`.

**Nota sobre o algoritmo:** este Thompson Sampling é **contextual** (regressão linear Bayesiana
por braço, usa as features do cliente), não o Beta-Bernoulli de 2 braços sem contexto que o plano
sugere como opção mais simples. O prior é `Normal(0, alpha² · I)` sobre os coeficientes de cada
braço, com `alpha=1.0` — o equivalente não-informativo ao `Beta(1,1)`: antes de ver dados, nenhum
braço é favorecido.

<!-- TODO Adryen: gráfico de conversão acumulada, análise exploração × explotação -->

## 6. Golden Set

<!-- responsável: Bertelli -->

5 clientes do split de teste (últimos 20%, fora do que o bandit usou pra treinar), um por perfil
sugerido no plano. `arm` é o canal que o banco realmente usou historicamente; `arm_recomendado` é
o que o Thompson treinado escolheria hoje (pela média da posterior de cada braço, sem sortear —
resultado reprodutível).

| Perfil | Canal real (histórico) | Canal recomendado | Justificativa |
|---|---|---|---|
| Jovem, sucesso em contato anterior | cellular | **telephone** | Alta propensão esperada (`poutcome=success`), mas o bandit não priorizou `cellular` para este cliente — contraintuitivo, ver "Limitações" abaixo. |
| Aposentado, nunca contatado | cellular | cellular | Aposentados convertem acima da média mesmo sem contato prévio; recomendação bate com a expectativa. |
| Blue-collar, casado, com empréstimo | cellular | cellular | Perfil historicamente menos propenso; o bandit ainda assim pende para `cellular` (taxa geral do canal é maior). |
| Saturado de contatos (campaign ≥ 6) | cellular | cellular | Muitos contatos sem sinal de conversão; recomendação segue o canal de maior taxa histórica. |
| Muitos campos "unknown" | cellular | cellular | Testa robustez do pipeline com dado incompleto — o preprocessor (`handle_unknown="ignore"`) não quebra. |

Gerar/atualizar esta tabela: `uv run python -m bm.golden_set`.

**Atenção — vale revisar antes da apresentação:** no primeiro caso ("alta propensão"), o bandit
recomendou o canal *oposto* ao que a intuição sugeria. Isso pode ser (a) ruído de uma seed
específica, (b) o bandit contextual captando um sinal real que a EDA não separou, ou (c) sintoma do
bug do replay já corrigido em `train.py` ainda deixar a política pouco decidida para casos fora do
padrão. Vale rodar `run_thompson_replay` de novo com outras seeds e comparar antes de afirmar
qualquer coisa sobre esse cliente no vídeo.

## 7. Arquitetura em nuvem

<!-- responsável: Matheus -->
<!-- TODO: mapeamento 1:1 dos artefatos do repo para serviços gerenciados -->

## 8. MLOps (ciclo de vida)

<!-- responsável: Bertelli -->

### Tracking compartilhado

Os runs do grupo vão para um **servidor MLflow no VPS** — os quatro enxergam os mesmos
experimentos. Para conectar:

```bash
cp .env.example .env     # preencher com a URL e a senha (peça ao Gabriel)
```

Feito isso, nada mais muda: `uv run python -m src.data_prep` já loga no servidor e
`uv run mlflow-ui` abre a UI compartilhada. **Sem `.env`, tudo funciona offline** em
`mlruns/mlflow.db` (fallback local, só na sua máquina).

Como subir o servidor: [`deploy/README.md`](deploy/README.md).

> **O `mlflow.db` nunca é commitado.** SQLite é binário e o git não sabe mesclar — dois colegas
> rodando experimentos no mesmo dia gerariam um merge que sobrescreve os runs de um deles.
> Compartilhamento se faz com servidor, não com repositório.

### Detalhes de configuração

O MLflow é configurado **sempre** via `src.tracking.setup_mlflow(experimento)`. Duas armadilhas que
isso resolve:

- O store padrão do MLflow é **relativo ao diretório de execução** — um script rodado da raiz e um
  notebook rodado de `notebooks/` gravariam em bancos diferentes. `setup_mlflow` ancora o store na
  raiz do repositório.
- O **MLflow 3 descontinuou o file store `mlruns/`** (que o plano original previa) e recusa o
  backend de filesystem. Usamos SQLite. Banco e artefatos ficam ambos sob `mlruns/`
  (`mlruns/mlflow.db` + `mlruns/artifacts/`), fora do git.

A Etapa 2 já loga (parâmetros da limpeza, taxa-base, conversão por braço, `preprocessor.joblib` e
`bandit_frame.parquet` como artefatos). Ver a UI com:

```bash
uv run mlflow-ui        # localhost:5000 — ou `uv run mlflow-ui 5001` para outra porta
```

### Ciclo de vida (dados → experimento → produção → feedback)

O ciclo completo deste projeto: os **dados** ficam versionados em `data/` (raw e processado,
com o contrato de `data_prep.py` garantindo que todo mundo aplica a mesma transformação); cada
rodada de simulação do bandit vira um **experimento rastreado** no MLflow
(`src/bm/mlflow_logging.py`), com os parâmetros (política, priors, epsilon, seed) e as métricas
(conversão, regret, N) registrados via `log_bandit_run`; o **estado do bandit** (a crença sobre
cada braço) é salvo como **artefato versionado** desse run; esse artefato é o que a **API
(Etapa 5)** carrega pra servir recomendações; cada resposta real de cliente (`POST /feedback`)
**realimenta o modelo**, fechando o loop online; e a **conversão por braço observada em
produção** é o sinal de **monitoramento** que indicaria a necessidade de um retreino ou reset —
se a conversão de um braço cair de forma sustentada, é sinal de que o comportamento do cliente
mudou (*drift*) e o bandit precisa reaprender, não só seguir ajustando incrementalmente.

**Ainda pendente:** instrumentar os runs reais da Etapa 3 com `log_bandit_run` assim que o
`train_bandit` do Adryen estiver pronto (logar `n_arms`, priors/epsilon, seed, conversão, regret
e `n_matched` de cada rodada).

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
mlruns/
  mlflow.db                     # MLflow: banco sqlite + artefatos (não versionado)
  artifacts/
```

`src/` existe justamente para que a API, a simulação do bandit e a avaliação chamem **as mesmas**
funções de limpeza/encoding — em vez de duplicar a lógica do notebook e criar train/serving skew.
