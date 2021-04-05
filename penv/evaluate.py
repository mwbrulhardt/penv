
import json

import click
import numpy as np
import ray
import ray.rllib.agents.ppo as ppo

from .env import create_env_with_price_series
from .price_generators import (
    MultiGBM,
    MultiSinePriceCurves
)


@click.command()
@click.option("--price-type", default="gbm", type=click.Choice(["gbm", "sine"]))
@click.option("--random", is_flag=True)
def main(price_type: str, random: bool):

    params = json.load(open("data/trained_params.json", "r"))

    config = params["config"]
    config["num_workers"] = 1
    config["explore"] = False
    config["env_config"]["total_steps"] = int(1e3)

    checkpoint = params["checkpoints"][0][0]

    agent = ppo.PPOTrainer(env="TradingEnv", config=config)
    agent.restore(checkpoint)

    config = config["env_config"]

    # Choose price stream
    if price_type == "sine":
        price_stream=MultiSinePriceCurves(
            s0=np.array([50, 48, 45, 60]),
            shift=np.array([0, np.pi / 2, np.pi, 3*np.pi / 2]),
            freq=np.array([1, 5, 3, 2]),
            n=config["total_steps"],
            warmup=config["min_periods"]
        )

    elif price_type == "gbm":
        price_stream = MultiGBM(
            s0=np.array([50, 48, 45, 60]),
            drift=np.array([0.05, 0.09, 0.04, 0.03]),
            volatility=np.array([1, 0.5, 0.3, 0.2]),
            rho=np.array([
                [ 1.        , -0.47945021, -0.28726099, -0.34183783],
                [-0.47945021,  1.        ,  0.15933449,  0.22425269],
                [-0.28726099,  0.15933449,  1.        ,  0.4950543 ],
                [-0.34183783,  0.22425269,  0.4950543 ,  1.        ]]
            ),
            n=config["total_steps"]
        )

    env = create_env_with_price_series(
        config=config,
        price_stream=price_stream
    )
    done = False
    obs = env.reset()
    action = env.action_scheme.weights.copy()

    while not done:
        if not random:
            action = agent.compute_action(obs, prev_action=action)
        else:
            action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

    env.render()


if __name__ == "__main__":
    ray.init()

    try:
        main()
    finally:
        ray.shutdown()
