#!/bin/sh
set -e

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
