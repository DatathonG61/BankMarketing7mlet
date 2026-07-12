#!/bin/sh
set -e

# Validacao explicita aqui, e nao com ${VAR:?} no compose: fixar valor no compose faz o Coolify
# marcar a variavel como somente-leitura no painel. Assim elas ficam editaveis e, ainda assim, o
# container se recusa a subir sem elas -- o que importa, porque um MLflow sem senha exposto na
# internet pode ser lido e apagado por qualquer um.
faltando=""
for v in MLFLOW_ADMIN_USERNAME MLFLOW_ADMIN_PASSWORD MLFLOW_FLASK_SERVER_SECRET_KEY; do
  eval "valor=\$$v"
  [ -z "$valor" ] && faltando="$faltando $v"
done

if [ -n "$faltando" ]; then
  echo "ERRO: variavel(is) obrigatoria(s) nao definida(s):$faltando" >&2
  echo "" >&2
  echo "Defina em Environment Variables no painel do Coolify:" >&2
  echo "  MLFLOW_ADMIN_USERNAME           ex: datathon" >&2
  echo "  MLFLOW_ADMIN_PASSWORD           uma senha forte" >&2
  echo "  MLFLOW_FLASK_SERVER_SECRET_KEY  gere com: openssl rand -hex 32" >&2
  exit 1
fi

# O basic-auth do MLflow le o usuario admin de um .ini, nao de variaveis de ambiente. Geramos o
# arquivo no boot a partir das env vars do Coolify, para a senha nunca ficar na imagem nem no git.
if [ ! -f "$MLFLOW_AUTH_CONFIG_PATH" ]; then
  cat > "$MLFLOW_AUTH_CONFIG_PATH" <<INI
[mlflow]
default_permission = READ
database_uri = sqlite:////mlflow/auth.db
admin_username = ${MLFLOW_ADMIN_USERNAME}
admin_password = ${MLFLOW_ADMIN_PASSWORD}
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
INI
  echo "auth.ini gerado para o usuario '${MLFLOW_ADMIN_USERNAME}'"
fi

exec "$@"
