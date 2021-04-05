
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from .model import ReallocationModel, Dirichlet
from .env import create_env


register_env("TradingEnv", create_env)
ModelCatalog.register_custom_action_dist("dirichlet", Dirichlet)
ModelCatalog.register_custom_model("reallocate", ReallocationModel)
