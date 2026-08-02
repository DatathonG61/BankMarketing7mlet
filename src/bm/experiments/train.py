import time
import mlflow
from src.bm.data_prep import CONTEXT_FEATURES

def train_bandit(bandit, train_data, preprocessor):
    X = preprocessor.transform(train_data[CONTEXT_FEATURES])

    arms = train_data["arm"].to_numpy()
    rewards = train_data["reward"].to_numpy()

    cumulative_reward = 0
    correct_actions = 0

    start = time.time()

    size = len(train_data)
    for step in range(size):
        print("Rodando step %d de %d", step, size)

        customer = X[step]

        predicted_arm = bandit.select_arm(customer)

        historical_arm = arms[step]

        reward = rewards[step] if predicted_arm == historical_arm else 0

        bandit.update(
            predicted_arm,
            customer,
            reward,
        )

        cumulative_reward += reward

        if predicted_arm == historical_arm:
            correct_actions += 1

        mlflow.log_metric(
            "reward",
            reward,
            step=step,
        )

        mlflow.log_metric(
            "cumulative_reward",
            cumulative_reward,
            step=step,
        )

    print("Finishing train")

    mlflow.log_metric(
        "accuracy_of_actions",
        correct_actions / len(train_data),
    )

    mlflow.log_metric(
        "final_reward",
        cumulative_reward,
    )

    mlflow.log_metric(
        "training_time",
        time.time() - start,
    )

    return bandit