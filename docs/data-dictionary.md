# Dicionário de Dados

Duas tabelas, dois contratos distintos:

| | Arquivo | Shape | Papel |
|---|---|---|---|
| **Raw** | `data/raw/bank-additional-full.csv` | 41.188 × 21 | Entrada bruta do Kaggle. **Nunca consumida diretamente por modelo.** |
| **Processed** | `data/processed/bandit_frame.parquet` | 41.176 × 20 | Saída da Etapa 2. É o que bandit, API e golden set consomem. |

Fonte: [Bank Marketing — henriqueyamahata](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)
(campanhas reais de um banco português, **mai/2008 → nov/2010**). Separador `;`.

> **A ordem das linhas é cronológica e é significativa.** Não existe coluna de data — a única
> informação temporal é a **posição da linha**. Embaralhar destrói irrecuperavelmente a
> ordenação e quebra a simulação de replay da Etapa 3.

---

## 1. Raw — `bank-additional-full.csv`

### 1.1 Dados do cliente

| Coluna | Tipo | Domínio | Observação |
|---|---|---|---|
| `age` | int | 17 – 98 (mediana 38) | |
| `job` | cat (12) | `admin.`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown` | 330 `unknown` (0,8%) |
| `marital` | cat (4) | `divorced`, `married`, `single`, `unknown` | 80 `unknown` (0,2%) |
| `education` | cat (8) | `basic.4y`, `basic.6y`, `basic.9y`, `high.school`, `illiterate`, `professional.course`, `university.degree`, `unknown` | 1.731 `unknown` (4,2%) |
| `default` | cat (3) | `no`, `unknown`, `yes` | **8.597 `unknown` (20,9%)**. `yes` tem só **3 linhas** em toda a base — proxy praticamente vazio. |
| `housing` | cat (3) | `no`, `unknown`, `yes` | 990 `unknown` (2,4%) |
| `loan` | cat (3) | `no`, `unknown`, `yes` | 990 `unknown` — **sempre nas mesmas linhas de `housing`** |

> ⚠️ **`"unknown"` é ausente disfarçado.** A base não tem nenhum `NaN`: o dado faltante vem como a
> string `"unknown"`. **Não imputamos** — mantemos como categoria própria, porque "o cliente não
> informou" é informação (ver [decisões](#3-decisões-de-transformação-raw--processed)).

### 1.2 Contato da campanha atual

| Coluna | Tipo | Domínio | Observação |
|---|---|---|---|
| `contact` | cat (2) | `cellular`, `telephone` | **É o braço (ação) do bandit.** Conversão: `cellular` 14,74%, `telephone` 5,23%. |
| `month` | cat (10) | `mar`–`dec` | **Não há `jan` nem `fev`** na base. |
| `day_of_week` | cat (5) | `mon`–`fri` | Só dias úteis. |
| `duration` | int | 0 – 4.918 (segundos) | 🚨 **VAZAMENTO — proibido.** Ver abaixo. |
| `campaign` | int | 1 – 56 | Nº de contatos nesta campanha, incluindo o atual. |

> 🚨 **`duration` é vazamento e nunca pode ser feature.** É a duração da ligação, ou seja, só é
> conhecida **depois** que a ligação terminou — quando o desfecho já aconteceu. Correlação 0,40 com
> `y` (média 221s para `no` contra 553s para `yes`). O próprio enunciado avisa duas vezes; o plano
> chama de *"desclassificação moral imediata"*. **É removida em `clean()` e não existe no processed.**

### 1.3 Campanha anterior

| Coluna | Tipo | Domínio | Observação |
|---|---|---|---|
| `pdays` | int | 0 – 999 | ⚠️ **`999` é sentinela, não número** — ver abaixo. |
| `previous` | int | 0 – 7 | Nº de contatos em campanhas anteriores. |
| `poutcome` | cat (3) | `failure` (4.252), `nonexistent` (35.551), `success` (1.373) | Desfecho da campanha anterior. |

> ⚠️ **`pdays == 999` significa "nunca contatado antes"**, não "999 dias atrás". Ocorre em **96,3%**
> das linhas. Usar como número puxaria a média para 962 e destruiria qualquer modelo. É convertido
> na flag booleana `was_contacted_before` e o `pdays` cru é **descartado**.

### 1.4 Contexto socioeconômico

Cinco indicadores macro do período. **Não descrevem o cliente** — descrevem *quando* a ligação
aconteceu. São idênticos para todos os clientes contatados na mesma data.

> 📌 **`month` não tem ano.** Um mesmo `may` agrega maio de 2008, 2009 e 2010 — por isso os
> indicadores trimestrais têm 2–3 valores distintos por nome de mês, e o `euribor3m` (diário) chega
> a 51. Na prática, **o par (`month`, macro) é o relógio da base**: é a única forma de recuperar o
> ano de uma linha. Reforça por que a ordem cronológica das linhas é a informação temporal real.

| Coluna | Tipo | Faixa | Observação |
|---|---|---|---|
| `emp.var.rate` | float | −3,4 – 1,4 | Taxa de variação do emprego (trimestral) |
| `cons.price.idx` | float | 92,2 – 94,8 | Índice de preços ao consumidor (mensal) |
| `cons.conf.idx` | float | −50,8 – −26,9 | Índice de confiança do consumidor (mensal) |
| `euribor3m` | float | 0,63 – 5,05 | Euribor 3 meses (diária) |
| `nr.employed` | float | 4.963,6 – 5.228,1 | Nº de empregados (trimestral, milhares) |

> ⚠️ **Altamente colineares entre si (0,91 – 0,97).** `emp.var.rate`, `euribor3m` e `nr.employed`
> codificam *quando* a campanha rodou (regime econômico 2008–2010), não quem é o cliente. Risco de o
> modelo aprender "a época" em vez de propensão. Mantidos como contexto, mas **sob vigilância**.

### 1.5 Target

| Coluna | Tipo | Domínio | Observação |
|---|---|---|---|
| `y` | cat (2) | `no` (88,73%), `yes` (11,27%) | Cliente assinou o depósito a prazo. **Desbalanceado ~8:1.** |

> **Acurácia é métrica inútil aqui** — chutar sempre `no` dá 88,7%. Usar conversão, regret, PR-AUC.

---

## 2. Processed — `bandit_frame.parquet`

Saída de `src.data_prep.build_bandit_frame()`. **41.176 linhas × 20 colunas** (12 duplicatas exatas
removidas). Ordem cronológica preservada linha a linha.

### 2.1 As duas colunas que definem o problema

| Coluna | Tipo | Domínio | Origem |
|---|---|---|---|
| `arm` | cat (2) | `cellular`, `telephone` | = `contact`. **A ação** que o bandit escolhe. |
| `reward` | int | 0 / 1 | = `(y == "yes")`. **A recompensa.** Taxa-base: **11,27%**. |

> **`arm` não é feature de contexto.** Ele é a *ação*, não um atributo do cliente — por isso está
> fora de `CONTEXT_FEATURES`. Incluí-lo daria ao modelo a própria decisão que ele deve recomendar.

### 2.2 Features de contexto (18)

Consumidas pelo `preprocessor.joblib` e pela variante contextual do bandit.

**Categóricas (9)** → one-hot, `handle_unknown="ignore"`:
`job`, `marital`, `education`, `default`, `housing`, `loan`, `month`, `day_of_week`, `poutcome`

**Numéricas (8)** → padronizadas (`StandardScaler`):
`age`, `campaign`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`

**Booleanas (1)** → passthrough:

| Coluna | Tipo | Origem |
|---|---|---|
| `was_contacted_before` | bool | `pdays != 999`. `False` em 39.661 linhas (96,3%), `True` em 1.515. |

Após o encoding: **18 colunas → 59 features**. O `"unknown"` sobrevive como coluna própria
(`cat__job_unknown`, `cat__default_unknown`, …), que é exatamente a intenção.

### 2.3 O que **não** existe no processed

| Removido | Motivo |
|---|---|
| `duration` | Vazamento — só conhecido após a ligação. |
| `pdays` (cru) | Sentinela 999; substituído por `was_contacted_before`. |
| `y` | Substituído por `reward` (0/1). |
| `contact` (como feature) | Vira `arm` — é ação, não contexto. |
| 12 linhas | Duplicatas exatas. |

---

## 3. Decisões de transformação (raw → processed)

| # | Decisão | Por quê |
|---|---|---|
| 1 | **Remover `duration`** | Vazamento. Não negociável. |
| 2 | **`pdays == 999` → `was_contacted_before`** | Sentinela, não número. |
| 3 | **Manter `"unknown"` como categoria** | É ausente *informativo* ("cliente não quis responder"), não ruído. Imputar apagaria o sinal. |
| 4 | **Remover 12 duplicatas exatas** | Registros idênticos inflariam a confiança do bandit. |
| 5 | **NÃO rebalancear** (sem SMOTE/over/under) | O bandit precisa da **taxa-base real** (11,27%) e das taxas reais por braço (14,74% / 5,23%). Reamostrar tornaria conversão, regret e replay pura ficção. O desbalanceamento é tratado com `class_weight="balanced"` e métricas adequadas — nunca alterando a distribuição. |
| 6 | **Split temporal, sem shuffle** | 80% inicial = simulação; 20% final = golden set. Embaralhar quebraria a lógica online do replay. |

---

## 4. ⚠️ Drift: treino e golden set são regimes econômicos diferentes

Consequência direta do split temporal. Não é bug — é a história do dataset, e precisa ser
considerada ao ler qualquer métrica do golden set.

| | Treino (80% inicial) | Golden set (20% final) |
|---|---|---|
| `euribor3m` | 4,27 | **1,03** |
| `emp.var.rate` | 0,65 | **−2,19** |
| `nr.employed` | 5.195 | 5.055 |
| **conversão** | **6,38%** | **30,83%** |

A Euribor despenca de 4,3% para 1,0% (crash de 2008 → corte de juros) e a conversão do depósito a
prazo quase **quintuplica**: sem concorrência de outras aplicações, o produto ficou atraente.

**Duas implicações:**

1. **É o argumento a favor do bandit.** Um classificador batch treinado nos primeiros 80% aprenderia
   "conversão ≈ 6%" e chegaria em 2010 prevendo 6% num mundo de 31% — descalibrado. O bandit se
   readapta a cada feedback.
2. **É uma limitação a declarar.** O golden set **não** é amostra representativa da base: qualquer
   métrica nele parecerá excelente pelo motivo errado (o período é fácil, não o modelo é bom).

---

## 5. Como reproduzir

```bash
uv run python -m src.data_prep     # raw → bandit_frame.parquet + preprocessor.joblib (log no MLflow)
```

```python
from src.data_prep import load_raw, clean, build_bandit_frame, temporal_split

frame = build_bandit_frame(clean(load_raw("data/raw/bank-additional-full.csv")))
train, golden = temporal_split(frame, test_size=0.2)   # sem shuffle
```

Inspecionar o encoder salvo:

```python
import joblib
prep = joblib.load("models/preprocessor.joblib")
prep.get_feature_names_out()                      # as 59 features
prep.named_transformers_["cat"].categories_       # categorias memorizadas
prep.named_transformers_["num"].mean_             # médias do TREINO
```
