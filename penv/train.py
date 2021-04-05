
import json
import os

import click
import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.stopper import TrialPlateauStopper

from .metrics import sharpe, maximum_drawdown


def on_episode_end(info):
    env = info["env"].vector_env.envs[0]
    history = env.observer.renderer_history
    pv = np.array([row["pv"] for row in history])
    returns = pd.Series(pv).pct_change()

    episode = info["episode"]
    episode.custom_metrics["sharpe"] = sharpe(returns.values)
    episode.custom_metrics["MDD"] = maximum_drawdown(pv)


@click.command()
@click.option("--workers", default=8, type=int)
def main(workers: int):
    params = json.load(open("data/tuned_params.json", "r"))

    config = params["config"].copy()
    config["num_workers"] = workers
    config["callbacks"] = {
        "on_episode_end": on_episode_end
    }
    checkpoint = params["checkpoints"][0][0]


    analysis = tune.run(
        "PPO",
        name="portfolio_allocation",
        config=config,
        stop={
            "episode_reward_mean": 70,
            "training_iteration": 150
        },
        restore=checkpoint,
        checkpoint_at_end=True,
        local_dir="./results"
    )

    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial(metric="episode_reward_min", mode="max"),
        metric="episode_reward_mean"
    )

    params["checkpoints"] = checkpoints
    json.dump(params, open("data/trained_params.json", "w"), indent=4)


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
