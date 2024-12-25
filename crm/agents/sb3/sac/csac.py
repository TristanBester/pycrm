from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    RolloutReturn,
    Schedule,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.sac.policies import (
    SACPolicy,
)

from crm.agents.sb3.vec import DispatchSubprocVecEnv


class CounterfactualSAC(SAC):
    """Counterfactual SAC implementation."""

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        """Initialize the SAC algorithm."""
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        # Check if the environment supports subprocess dispatching
        self.subproc_dispatch_supported = isinstance(self.env, DispatchSubprocVecEnv)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "C-SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "CounterfactualSAC":
        """Override to change algorithm Tensorboard log name."""
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: ActionNoise | None = None,
        learning_starts: int = 0,
        log_interval: int | None = None,
    ) -> RolloutReturn:
        """Collect experiences and store them into a ReplayBuffer.

        Args:
            env: The training environment.
            callback: Callback that will be called at each step
                (and at the beginning and end of the rollout).
            train_freq: How much experience to collect
                by doing rollouts of current policy.
                Either TrainFreq(<n>, TrainFrequencyUnit.STEP)
                or TrainFreq(<n>, TrainFrequencyUnit.EPISODE)
                with <n> being an integer greater than 0.
            action_noise: Action noise that will be used for exploration.
                Required for deterministic policy (e.g. TD3). This can also be used
                in addition to the stochastic policy for SAC.
            learning_starts: Number of steps before learning for the warm-up phase.
            replay_buffer: The buffer to store experiences.
            log_interval: Log data every log_interval episodes.

        Returns:
            None
        """
        # Switch to eval mode (this affects batch norm / dropout)
        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."
        if env.num_envs > 1:
            assert (
                train_freq.unit == TrainFrequencyUnit.STEP
            ), "You must use only one env when doing episodic training."

        self.policy.set_training_mode(False)
        if self.use_sde:
            self.actor.reset_noise(env.num_envs)
        num_collected_steps, num_collected_episodes = 0, 0

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(
            train_freq, num_collected_steps, num_collected_episodes
        ):
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and num_collected_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(
                learning_starts, action_noise, env.num_envs
            )

            # Rescale and perform action
            new_obs, _, dones, infos = env.step(actions)
            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(
                    num_collected_steps * env.num_envs,
                    num_collected_episodes,
                    continue_training=False,
                )

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)
            self._store_counterfactual_transitions(
                replay_buffer, buffer_actions, new_obs
            )
            self._update_current_progress_remaining(
                self.num_timesteps, self._total_timesteps
            )

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()
            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = {"indices": [idx]} if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if (
                        log_interval is not None
                        and self._episode_num % log_interval == 0
                    ):
                        self._dump_logs()
        callback.on_rollout_end()
        return RolloutReturn(
            num_collected_steps * env.num_envs,
            num_collected_episodes,
            continue_training,
        )

    def _store_counterfactual_transitions(
        self,
        replay_buffer: ReplayBuffer,
        buffer_actions,
        obs_next,
    ) -> None:
        assert isinstance(self.env, VecEnv), "You must pass a VecEnv"

        if self.subproc_dispatch_supported:
            assert isinstance(
                self.env, DispatchSubprocVecEnv
            ), "You must pass a DispatchSubprocVecEnv"

            # Get ground observations
            ground_obs = self.env.dispatched_env_method("to_ground_obs", self._last_obs)
            ground_obs_next = self.env.dispatched_env_method("to_ground_obs", obs_next)

            # Generate counterfactual experience
            result = self.env.dispatched_env_method(
                "generate_counterfactual_experience",
                ground_obs,
                buffer_actions,
                ground_obs_next,
            )
        else:
            # Get ground observations
            ground_obs = self.env.env_method("to_ground_obs", self._last_obs[0])  # type: ignore
            ground_obs_next = self.env.env_method("to_ground_obs", obs_next[0])  # type: ignore

            # Generate counterfactual experience
            result = self.env.env_method(
                "generate_counterfactual_experience",
                ground_obs[0],
                buffer_actions[0],
                ground_obs_next[0],
            )

        c_obs, c_actions, c_obs_next, c_rewards, c_dones, c_infos = zip(
            *result, strict=True
        )
        c_obs = np.concatenate(c_obs)
        c_actions = np.concatenate(c_actions)
        c_obs_next = np.concatenate(c_obs_next)
        c_rewards = np.concatenate(c_rewards)
        c_dones = np.concatenate(c_dones)
        c_infos = np.concatenate(c_infos)

        # Reshape & batch to match number of envs
        c_obs = self.reshape_and_trim(
            c_obs,
            final_dim=self.env.observation_space.shape[0],  # type: ignore
        )
        c_actions = self.reshape_and_trim(
            c_actions,
            final_dim=self.env.action_space.shape[0],  # type: ignore
        )
        c_obs_next = self.reshape_and_trim(
            c_obs_next,
            final_dim=self.env.observation_space.shape[0],  # type: ignore
        )
        c_rewards = self.reshape_and_trim(c_rewards, final_dim=1)
        c_dones = self.reshape_and_trim(c_dones, final_dim=1)
        c_infos = self.reshape_and_trim(c_infos, final_dim=1)

        # Insert counterfactual transitions into replay buffer
        for i in range(len(c_obs)):
            replay_buffer.add(
                obs=c_obs[i],
                next_obs=c_obs_next[i],
                action=c_actions[i],
                reward=c_rewards[i],
                done=c_dones[i],
                infos=c_infos[i],
            )

        self._last_obs = obs_next

    def reshape_and_trim(self, array, final_dim):
        """Trim into batches to match number of environments."""
        assert isinstance(self.env, VecEnv), "You must pass a VecEnv"

        # Desired shape
        if final_dim > 1:
            target_shape = (-1, self.env.num_envs, final_dim)
        else:
            target_shape = (-1, self.env.num_envs)
        num_elements_per_batch = self.env.num_envs * final_dim

        # Flatten the array to make slicing easier
        flat_array = array.flatten()

        # Calculate the number of elements required
        num_elements_required = num_elements_per_batch * (
            len(flat_array) // num_elements_per_batch
        )

        # Trim the array to fit the required number of elements
        trimmed_array = flat_array[:num_elements_required]

        # Reshape the array to the desired shape
        reshaped_array = trimmed_array.reshape(target_shape)
        return reshaped_array
