import os

import hydra
import wandb
from omegaconf import DictConfig
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from crm.agents.sb3.vec import DispatchSubprocVecEnv
from experiments.warehouse.lib.agents import LoggingCounterfactualSAC


@hydra.main(
    config_path="../conf",
    config_name="config",
    version_base=None,
)
def main(config: DictConfig) -> None:
    """Main function."""
    method_name = (
        f"C-SAC_{config.exp.control_type}_{config.exp.name}_{config.train.seed}"
    )
    if config.exp.use_wandb:
        wandb.init(
            project=config.exp.wandb_project,
            name=method_name,
            tags=[
                "sac",
                config.exp.control_type,
                config.exp.name,
            ],
            sync_tensorboard=True,
        )

    vec_env = make_vec_env(
        "Warehouse-v0",
        n_envs=config.train.n_procs,
        vec_env_cls=DispatchSubprocVecEnv,
        seed=config.train.seed,
        env_kwargs={
            "ground_env_kwargs": {
                "control_type": config.exp.control_type,
            },
            "crm_kwargs": {},
            "lf_kwargs": {},
            "crossproduct_kwargs": {
                "max_steps": config.exp.max_steps,
            },
        },
    )

    model = LoggingCounterfactualSAC(
        "MlpPolicy",
        vec_env,
        verbose=config.train.verbose,
        tensorboard_log="logs/",
        seed=config.train.seed,
        device=config.hparams.device,
        ent_coef=config.hparams.ent_coef,
        buffer_size=config.hparams.buffer_size,
        batch_size=config.hparams.batch_size,
        learning_rate=config.hparams.learning_rate,
        tau=config.hparams.tau,
        gradient_steps=config.hparams.gradient_steps,
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
