import time
import mlflow
from src.bm.data_prep import CONTEXT_FEATURES

def train_bandit(bandit, train_data, preprocessor, log_to_mlflow=True):
    """Roda o replay sobre `train_data` e devolve `(bandit, rewards_matched)`.

    `rewards_matched` é a sequência (na ordem cronológica) das recompensas das linhas
    onde o braço escolhido bateu com o histórico -- é o que `bm.evaluation` usa pra
    calcular conversão e regret. `log_to_mlflow=False` pula o log por linha (usado pela
    varredura de seeds da Etapa 7.1, senão vira uma chamada ao MLflow por linha vezes
    N seeds).
    """
    X = preprocessor.transform(train_data[CONTEXT_FEATURES])

    arms = train_data["arm"].to_numpy()
    rewards = train_data["reward"].to_numpy()

    cumulative_reward = 0
    correct_actions = 0
    rewards_matched = []

    start = time.time()

    size = len(train_data)
    for step in range(size):
        if log_to_mlflow and step % 5000 == 0:
            print(f"Rodando step {step} de {size}")

        customer = X[step]

        predicted_arm = bandit.select_arm(customer)
        historical_arm = arms[step]

        if predicted_arm != historical_arm:
            # Replay / rejection sampling (Li et al., 2011): só sabemos o resultado do
            # braço que de fato foi usado no histórico. Se o bandit escolheu outro
            # braço, a linha é descartada -- não conta e não atualiza o bandit.
            continue

        reward = rewards[step]
        bandit.update(
            predicted_arm,
            customer,
            reward,
        )

        cumulative_reward += reward
        correct_actions += 1
        rewards_matched.append(reward)

        if log_to_mlflow:
            mlflow.log_metric("reward", reward, step=step)
            mlflow.log_metric("cumulative_reward", cumulative_reward, step=step)

    if log_to_mlflow:
        print("Finishing train")
        mlflow.log_metric("accuracy_of_actions", correct_actions / size)
        mlflow.log_metric("final_reward", cumulative_reward)
        mlflow.log_metric("training_time", time.time() - start)

    return bandit, rewards_matched