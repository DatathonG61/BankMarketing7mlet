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
uv run pytest -q                          # testes
uv run uvicorn src.api:app --reload       # sobe a API (docs em /docs)
```

Requer Python >= 3.13. O CSV bruto já está em `data/raw/`; se faltar, o notebook de EDA o baixa do
Kaggle automaticamente (requer `~/.kaggle/kaggle.json`).

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
<!-- TODO: MLflow — priors, epsilon, seed e métricas logados; ciclo dados → experimento →
     estado do bandit → serving → feedback → monitoramento → reset em drift -->

## 9. Limitações

<!-- responsável: todos -->
<!-- TODO: avaliação offline por replay ≠ produção; dados de 2008–2010 -->

---

## Estrutura do repositório

```
data/raw/            # bank-additional-full.csv (commitado)
data/processed/      # base tratada + bandit_frame.parquet (saída da Etapa 2)
notebooks/
  01-eda.ipynb       # EDA + análise de leakage + definição dos braços (Etapa 1)
  01-baseline.ipynb  # baseline de propensão
src/
  data_prep.py       # load_raw() / clean() / build_bandit_frame() — contrato de dados
  bandit.py          # EpsilonGreedy / ThompsonSampling — select_arm() / update()
  api.py             # FastAPI: /recommend, /feedback, /health
models/              # preprocessor.joblib, bandit_state.json
reports/figures/     # gráficos usados no README e no vídeo
tests/               # pytest
mlruns/              # MLflow (não versionado)
```

`src/` existe justamente para que a API, a simulação do bandit e a avaliação chamem **as mesmas**
funções de limpeza/encoding — em vez de duplicar a lógica do notebook e criar train/serving skew.
