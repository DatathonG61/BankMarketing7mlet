FROM python:3.13-slim

# A versao deve bater com a do pyproject (mlflow==3.14.0). Cliente e servidor com schemas
# diferentes produzem erros obscuros.
# O extra [auth] traz o Flask-WTF, exigido pelo --app-name basic-auth (CSRF).
RUN pip install --no-cache-dir "mlflow[auth]==3.14.0"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /mlflow
EXPOSE 5000

ENTRYPOINT ["/entrypoint.sh"]
