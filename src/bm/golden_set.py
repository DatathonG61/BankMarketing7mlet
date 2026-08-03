"""Seleção e recomendação dos 5 clientes do golden set — Etapa 4.2 (Bertelli).

Escolhe, dentro do `bandit_frame` (saída de `data_prep.build_bandit_frame`), uma linha
real para cada um dos 5 perfis sugeridos no plano do grupo, e -- agora que o bandit da
Etapa 3 (Adryen) existe -- calcula a recomendação de cada um.

Perfis (ver `Plano 7MLET Refinado.pdf`, seção 4.2):
1. jovem_sucesso_previo — alta propensão (contato anterior converteu)
2. aposentado_nunca_contatado — propensão média-alta
3. blue_collar_com_emprestimos — propensão historicamente baixa
4. saturado_de_contatos — cliente já contatado muitas vezes nesta campanha
5. muitos_unknown — dado faltante disfarçado, testa robustez
"""

import pandas as pd

from bm.data_prep import CONTEXT_FEATURES

JUSTIFICATIVAS: dict[str, str] = {
    "jovem_sucesso_previo": (
        "já converteu numa campanha anterior (poutcome=success) — perfil de alta "
        "propensão; comparar com a recomendação real do bandit e discutir se bate "
        "com a expectativa (ver nota no README)"
    ),
    "aposentado_nunca_contatado": (
        "aposentados convertem acima da média neste dataset mesmo sem contato prévio "
        "— propensão média-alta"
    ),
    "blue_collar_com_emprestimos": (
        "perfil (profissão + já tem empréstimo) historicamente menos propenso a "
        "fechar depósito a prazo"
    ),
    "saturado_de_contatos": (
        "6+ contatos nesta campanha sem sinal de conversão — caso de saturação, "
        "candidato a parar de insistir"
    ),
    "muitos_unknown": (
        "vários campos 'unknown' — testa se o pipeline (one-hot com "
        "handle_unknown='ignore') segura um cliente com dado incompleto sem quebrar"
    ),
}


def select_golden_set_clients(frame: pd.DataFrame) -> pd.DataFrame:
    """Devolve até 5 linhas de `frame`, uma por perfil, marcadas na coluna `perfil_golden_set`.

    Se algum perfil não tiver nenhuma linha correspondente na base, ele simplesmente não
    aparece no resultado (documentar essa ausência no README, se acontecer).
    """
    perfis = {
        "jovem_sucesso_previo": (frame["age"] <= 30) & (frame["poutcome"] == "success"),
        "aposentado_nunca_contatado": (frame["job"] == "retired")
        & (~frame["was_contacted_before"]),
        "blue_collar_com_emprestimos": (frame["job"] == "blue-collar")
        & (frame["marital"] == "married")
        & ((frame["housing"] == "yes") | (frame["loan"] == "yes")),
        "saturado_de_contatos": frame["campaign"] >= 6,
        "muitos_unknown": (
            frame[["job", "marital", "education", "default", "housing", "loan"]] == "unknown"
        ).sum(axis=1)
        >= 2,
    }

    linhas = []
    for nome_perfil, mascara in perfis.items():
        candidatos = frame[mascara]
        if candidatos.empty:
            continue
        linha = candidatos.iloc[[0]].copy()
        linha["perfil_golden_set"] = nome_perfil
        linhas.append(linha)

    if not linhas:
        return frame.iloc[0:0]
    return pd.concat(linhas, ignore_index=True)


def recommend_for_golden_set(
    golden_frame: pd.DataFrame, bandit, preprocessor
) -> pd.DataFrame:
    """Para cada cliente do golden set, calcula o braço recomendado pelo bandit treinado.

    Usa a média da posterior de cada braço (`a_inv @ b`, sem amostrar) em vez de
    `select_arm` -- é determinístico e reproduzível, o que faz mais sentido pra um
    relatório do que um sorteio que muda a cada chamada. `score_<braco>` é a
    pontuação estimada (não é uma probabilidade calibrada; serve pra comparar os
    braços entre si). Adiciona também a coluna `justificativa` com a explicação de
    uma frase de cada perfil (ver `JUSTIFICATIVAS`).
    """
    X = preprocessor.transform(golden_frame[CONTEXT_FEATURES])

    recomendacoes = []
    for x in X:
        scores = {
            arm: float((state.a_inv @ state.b) @ x) for arm, state in bandit.arms.items()
        }
        recomendacoes.append(
            {"arm_recomendado": max(scores, key=scores.get), **scores}
        )

    out = golden_frame.reset_index(drop=True).join(pd.DataFrame(recomendacoes))
    out["justificativa"] = out["perfil_golden_set"].map(JUSTIFICATIVAS)
    return out


if __name__ == "__main__":
    import sys

    import joblib

    from bm.data_prep import temporal_split

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    frame_real = pd.read_parquet("data/processed/bandit_frame.parquet")
    # golden set vem do split de teste (últimos 20%, plano seção 2.3/4.2) -- não do
    # mesmo trecho usado pra treinar o bandit.
    _, teste = temporal_split(frame_real)

    preprocessor_real = joblib.load("models/preprocessor.joblib")
    bandit_real = joblib.load("models/thompson.joblib")

    clientes = select_golden_set_clients(teste)
    resultado = recommend_for_golden_set(clientes, bandit_real, preprocessor_real)

    colunas = [
        "perfil_golden_set",
        "arm",
        "arm_recomendado",
        "cellular",
        "telephone",
        "justificativa",
    ]
    print(resultado[colunas].to_string(index=False))
