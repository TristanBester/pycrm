import os
from warnings import filterwarnings

import hydra
import wandb
from omegaconf import DictConfig
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from crm.agents.sb3.vec import DispatchSubprocVecEnv
from experiments.warehouse.lib.agents.context_sensitive import (
    ContextSensitiveSubtaskLoggingCSAC,
)
from experiments.warehouse.lib.crossproducts.context_sensitive import (
    ContextSensitiveCrossProductMDP,
)

filterwarnings("ignore")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(config: DictConfig):
    method_name = f"CSAC_{config.exp.name}_{config.train.seed}"
    if config.exp.use_wandb:
        wandb.init(
            project=config.exp.wandb_project,
            name=method_name,
            tags=["csac", config.exp.name, "v5"],
            sync_tensorboard=True,
        )

    env = make_vec_env(
        ContextSensitiveCrossProductMDP,
        n_envs=config.train.n_procs,
        vec_env_cls=DispatchSubprocVecEnv,
        env_kwargs={"render_mode": "rgb_array"},
        seed=config.train.seed,
    )

    model = ContextSensitiveSubtaskLoggingCSAC(
        "MlpPolicy",
        env,
        verbose=config.train.verbose,
        tensorboard_log="logs/",
        seed=config.train.seed,
        device="cuda",
        ent_coef=config.hparams.ent_coef,
        buffer_size=config.hparams.buffer_size,
        batch_size=config.hparams.batch_size,
        learning_rate=config.hparams.learning_rate,
        tau=config.hparams.tau,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=config.train.checkpoint_interval,
        save_path=os.path.join(config.environment.checkpoint_dir, method_name),
        name_prefix="model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callback = CallbackList([checkpoint_callback])

    model.learn(
        total_timesteps=config.exp.total_timesteps,
        log_interval=config.exp.log_interval,
        tb_log_name=method_name,
        callback=callback,
    )


if __name__ == "__main__":
    raise SystemExit(main())
