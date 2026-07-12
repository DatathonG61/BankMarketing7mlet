# Servidor MLflow compartilhado (VPS + Coolify)

O objetivo é que os quatro vejam **os mesmos experimentos**. Os runs vão para um servidor MLflow no
VPS; o repositório continua sem nenhum banco versionado.

> **Por que não commitar o `mlflow.db`?** SQLite é binário e o git não sabe mesclar. Duas pessoas
> rodando experimentos no mesmo dia geram um merge que **sobrescreve os runs de uma delas**. E
> SQLite em pasta sincronizada (Google Drive/Dropbox) **corrompe** — o sync não respeita as travas
> de arquivo. Compartilhamento se faz com servidor, não com repositório.

O SQLite aqui é seguro porque **um único processo** (o servidor) é dono do arquivo e serializa as
escritas. Para 4 pessoas, sobra folga.

---

## Arquitetura

```
   você / Adryen / Bertelli / Matheus
              │  HTTPS + usuário/senha
              ▼
    ┌──────────────────────┐
    │  proxy do Coolify    │  domínio + TLS automático (o Coolify cuida)
    └──────────┬───────────┘
               │  rede interna do Docker
    ┌──────────▼───────────┐
    │  MLflow (:5000)      │  --app-name basic-auth  --serve-artifacts
    └──────────┬───────────┘
               │
        volume mlflow-data
        ├── mlflow.db     (runs, params, métricas)
        ├── auth.db       (usuários)
        └── artifacts/    (preprocessor.joblib, bandit_frame.parquet, gráficos)
```

O compose **não publica portas nem sobe proxy próprio** — isso brigaria com o proxy do Coolify pelas
portas 80/443. Ele só expõe a 5000 na rede interna e deixa o Coolify rotear.

---

## Deploy no Coolify

### 1. Criar o recurso

No painel do Coolify: **+ New** → **Docker Compose** → apontar para este repositório
(`DatathonG61/BankMarketing7mlet`), branch `develop`, com **Base Directory** = `/deploy`.

### 2. Definir as variáveis de ambiente

Em **Environment Variables**, criar as três (todas obrigatórias — o compose se recusa a subir sem):

| Variável | Valor |
|---|---|
| `MLFLOW_ADMIN_USERNAME` | `datathon` |
| `MLFLOW_ADMIN_PASSWORD` | uma senha forte — é a que o time vai usar |
| `MLFLOW_FLASK_SERVER_SECRET_KEY` | gere com `openssl rand -hex 32` |

O domínio **não** entra aqui — ele vai no campo Domains (passo 3).

> A `SECRET_KEY` precisa ser **fixa**. Se mudar a cada deploy, as sessões abertas do time são
> invalidadas. Gere uma vez e não mexa mais.

### 3. Domínio e HTTPS

O domínio se define no campo **Domains** do serviço `mlflow`, **com a porta interna no final**:

```
https://mlflow.<SEU-IP-COM-PONTOS>.sslip.io:5000
```

Dois detalhes que fazem toda a diferença:

- **`https://` no começo** — é o que faz o Traefik pedir o certificado ao Let's Encrypt. Sem ele,
  serve em HTTP puro, e aí o basic-auth trafega usuário e senha em base64, que é reversível.
- **`:5000` no final** — não é a porta pública (essa continua sendo a 443). É a porta **interna do
  container** para onde o Traefik encaminha. Sem ela o Coolify assume a 80, não acha ninguém
  escutando e devolve **503** (mesmo com o container saudável).

> ⚠️ **Não declare `SERVICE_FQDN_MLFLOW_5000` no compose nem crie essa variável no painel.** Essa
> "variável mágica" do Coolify é gerada e **travada** por ele: depois de criada não dá mais para
> editar, e ela sequestra o campo Domains (o que você digita ali é sobrescrito no save). Foi por
> isso que a removemos do `docker-compose.yml`. Se ela já existir no seu recurso, **apague-a** (junto
> com `SERVICE_URL_MLFLOW`) e faça um **Redeploy** completo — os labels do Traefik só são regerados
> no deploy, um Restart reaproveita os antigos.

**Sem domínio próprio?** Use o [sslip.io](https://sslip.io): ele resolve qualquer hostname que
contenha um IP para aquele IP, sem cadastro. Se o VPS é `203.0.113.45`, então
`mlflow.203.0.113.45.sslip.io` já aponta para ele. E como o `sslip.io` está na Public Suffix List, o
Let's Encrypt emite certificado normalmente — é HTTPS real.

**Pré-requisitos para o certificado sair:**
- portas **80 e 443** abertas no firewall (a 80 é obrigatória — é por ela que o Let's Encrypt valida);
- e-mail do Let's Encrypt configurado nas **Settings** do Coolify (é o passo mais esquecido).

Se depois de ~2 min o certificado ainda for `CN = TRAEFIK DEFAULT CERT` (autoassinado), é porque a
rota não foi registrada ou a validação falhou — confira os dois itens acima.

### 4. Deploy

Clicar em **Deploy**. Acompanhar os logs até aparecer `Uvicorn running`. Abrir a URL: deve pedir
usuário e senha.

---

## Para o time (cada um na sua máquina)

```bash
cp .env.example .env     # o .env.example da RAIZ do repo, não o desta pasta
```

Preencher com o que o Gabriel passar:

```ini
MLFLOW_TRACKING_URI=https://mlflow.seu-dominio.com
MLFLOW_TRACKING_USERNAME=datathon
MLFLOW_TRACKING_PASSWORD=a-senha-definida-no-coolify
```

Pronto. Nada mais muda: `uv run python -m src.data_prep` já loga no servidor e `uv run mlflow-ui`
abre a UI compartilhada.

**Sem `.env`, tudo continua funcionando offline** — os runs caem em `mlruns/mlflow.db`, só na sua
máquina. É o fallback, e é o que roda em CI.

---

## Notas de operação

- **Backup.** O banco é um arquivo só; o datathon inteiro cabe nele. Pelo terminal do Coolify:
  ```bash
  sqlite3 /mlflow/mlflow.db ".backup '/mlflow/backup.db'"
  ```
  Vale fazer isso antes da entrega.

- **Versão travada.** O `Dockerfile` fixa `mlflow[auth]==3.14.0`, a mesma do `pyproject.toml`.
  Cliente e servidor com schemas diferentes produzem erros obscuros — se subir a versão num lado,
  suba no outro.

- **O extra `[auth]` é obrigatório.** O `--app-name basic-auth` precisa do `Flask-WTF` (validação
  CSRF); com `pip install mlflow` puro o container sobe e morre em loop.

- **Usuários adicionais.** O `admin` criado no boot pode cadastrar outros pela API
  (`/api/2.0/mlflow/users/create`), mas para 4 pessoas o mais simples é todos usarem a mesma
  credencial.

- **Se preferir não expor nada na internet**, dá para pular o domínio e acessar por túnel SSH:
  `ssh -L 5000:localhost:5000 root@<IP>` e usar `MLFLOW_TRACKING_URI=http://localhost:5000`. Mais
  seguro, porém cada um precisa abrir o túnel antes de rodar.
