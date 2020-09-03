import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from model import HerdingModel
from utils.rllib_multiagent_adapter import MultiAgentHerding


env = MultiAgentHerding()
obs_space = env.env.observation_space
act_space = env.env.action_space

config = {

    "env": MultiAgentHerding,
    "model": {
        "custom_model": "herding_model"
    },
    "multiagent": {
        "policies": {
            "policy": (None, obs_space, act_space, {})
        },
        "policy_mapping_fn": lambda agent_id: "policy",
    },
    "horizon": 2000,
    "num_gpus": 1,
    #"num_workers": 4,
    #"num_envs_per_worker": 2,
    #"num_sgd_iter": 5,
    #"train_batch_size": 400,
    #"evaluation_num_episodes": 10,
    #"rollout_fragment_length": 20,
}

if __name__ == "__main__":
    ray.init()
    ModelCatalog.register_custom_model("herding_model", HerdingModel)

    stop = {
        "training_iteration": 10
    }
    results = tune.run(
        "PPO",
        name="Herding",
        stop=stop,
        config=config,
        checkpoint_at_end=True
    )

    ray.shutdown()
