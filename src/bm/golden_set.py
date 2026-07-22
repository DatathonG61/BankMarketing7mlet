"""Seleção dos 5 clientes do golden set — Etapa 4.2 (Bertelli).

Escolhe, dentro do `bandit_frame` (saída de `data_prep.build_bandit_frame`), uma linha
real para cada um dos 5 perfis sugeridos no plano do grupo. Só faz a *seleção* dos
clientes — a "recomendação" de cada um (o que o bandit escolheria) só existe depois que
o bandit estiver treinado (Etapa 3, Adryen), então isso fica pra depois, fora daqui.

Perfis (ver `Plano 7MLET Refinado.pdf`, seção 4.2):
1. jovem_sucesso_previo — alta propensão (contato anterior converteu)
2. aposentado_nunca_contatado — propensão média-alta
3. blue_collar_com_emprestimos — propensão historicamente baixa
4. saturado_de_contatos — cliente já contatado muitas vezes nesta campanha
5. muitos_unknown — dado faltante disfarçado, testa robustez
"""

import pandas as pd


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
