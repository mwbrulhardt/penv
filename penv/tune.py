
import json
import os

import click
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining


@click.command()
@click.option("--num-samples", default=4, type=int)
@click.option("--num-workers", default=8, type=int)
def main(num_samples: int, num_workers: int):
    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=50,
        resample_probability=0.25,
        hyperparam_mutations={
            "lambda": tune.uniform(0.9, 1.0),
            "clip_param": tune.uniform(0.01, 0.5),
            "lr": [1e-2, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "num_sgd_iter": tune.randint(1, 30),
            "sgd_minibatch_size": tune.randint(128, 16384),
            "train_batch_size": tune.randint(2000, 160000),
        }
    )

    analysis = tune.run(
        "PPO",
        name="pbt_portfolio_reallocation",
        scheduler=pbt,
        num_samples=num_samples,
        metric="episode_reward_min",
        mode="max",
        config={
            "env": "TradingEnv",
            "env_config":{
                "total_steps": 1000,
                "num_assets": 4,
                "commission": 1e-3,
                "time_cost": 0,
                "window_size": tune.randint(5, 50),
                "min_periods": 150
            },
            "kl_coeff": 1.0,
            "num_workers": num_workers,
            "num_gpus": 0,
            "observation_filter": tune.choice(["NoFilter", "MeanStdFilter"]),
            "framework": "torch",
            "model": {
                "custom_model": "reallocate",
                "custom_model_config": {
                   "num_assets": 4
                },
                "custom_action_dist": "dirichlet",
            },
            "num_sgd_iter": 10,
            "sgd_minibatch_size": 128,
            "lambda": tune.uniform(0.9, 1.0),
            "clip_param": tune.uniform(0.1, 0.5),
            "lr": tune.loguniform(1e-2, 1e-5),
            "train_batch_size": tune.randint(1000, 20000)
        },
        stop={
            "episode_reward_min": 20,
            "training_iteration": 100
        },
        checkpoint_at_end=True,
        local_dir="./results"
    )

    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial(metric="episode_reward_min", mode="max"),
        metric="episode_reward_mean"
    )

    params = {
        "config": analysis.best_config,
        "checkpoints": checkpoints
    }

    json.dump(params, open("data/tuned_params.json", "w"), indent=4)


if __name__ == "__main__":
    ray.init()

    main()

    ray.shutdown()
