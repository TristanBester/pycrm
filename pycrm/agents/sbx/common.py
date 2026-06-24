from typing import Any, cast

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import (
    RolloutReturn,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv

from pycrm.agents.sb3.wrapper import DispatchSubprocVecEnv


class CounterfactualOffPolicyMixin:
    """Replay-buffer counterfactual collection for SB3-compatible off-policy agents."""

    def _init_counterfactual_support(self) -> None:
        model = cast(Any, self)
        model.subproc_dispatch_supported = isinstance(
            model.env, DispatchSubprocVecEnv
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
        """Collect experience and insert CRM counterfactual transitions."""
        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."
        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, (
                "You must use only one env when doing episodic training."
            )

        model = cast(Any, self)
        if hasattr(model.policy, "set_training_mode"):
            model.policy.set_training_mode(False)

        use_sde = bool(getattr(model, "use_sde", False))
        actor = getattr(model, "actor", None)
        if use_sde and actor is not None:
            actor.reset_noise(env.num_envs)

        num_collected_steps, num_collected_episodes = 0, 0
        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(
            train_freq, num_collected_steps, num_collected_episodes
        ):
            sde_sample_freq = int(getattr(self, "sde_sample_freq", -1))
            if (
                use_sde
                and actor is not None
                and sde_sample_freq > 0
                and num_collected_steps % sde_sample_freq == 0
            ):
                actor.reset_noise(env.num_envs)

            actions, buffer_actions = model._sample_action(
                learning_starts, action_noise, env.num_envs
            )
            new_obs, _, dones, infos = env.step(actions)
            model.num_timesteps += env.num_envs
            num_collected_steps += 1

            callback.update_locals(locals())
            if not callback.on_step():
                return RolloutReturn(
                    num_collected_steps * env.num_envs,
                    num_collected_episodes,
                    continue_training=False,
                )

            model._update_info_buffer(infos, dones)
            self._store_counterfactual_transitions(
                replay_buffer, buffer_actions, new_obs, dones, infos
            )
            model._update_current_progress_remaining(
                model.num_timesteps, model._total_timesteps
            )
            model._on_step()

            for idx, done in enumerate(dones):
                if done:
                    num_collected_episodes += 1
                    model._episode_num += 1

                    if action_noise is not None:
                        kwargs = {"indices": [idx]} if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    if (
                        log_interval is not None
                        and model._episode_num % log_interval == 0
                    ):
                        model._dump_logs()

        callback.on_rollout_end()
        return RolloutReturn(
            num_collected_steps * env.num_envs,
            num_collected_episodes,
            continue_training,
        )

    def _store_counterfactual_transitions(
        self,
        replay_buffer: ReplayBuffer,
        buffer_actions: np.ndarray,
        obs_next: Any,
        dones: Any,
        infos: list[dict[str, Any]],
    ) -> None:
        model = cast(Any, self)
        assert isinstance(model.env, VecEnv), "You must pass a VecEnv"

        obs_next_terminal = obs_next.copy()
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                obs_next_terminal[i] = infos[i]["terminal_observation"]

        if model.subproc_dispatch_supported:
            assert isinstance(model.env, DispatchSubprocVecEnv), (
                "You must pass a DispatchSubprocVecEnv"
            )
            ground_obs = model.env.dispatched_env_method(
                "to_ground_obs", model._last_obs
            )
            ground_obs_next = model.env.dispatched_env_method(
                "to_ground_obs", obs_next_terminal
            )
            result = model.env.dispatched_env_method(
                "generate_counterfactual_experience",
                ground_obs,
                buffer_actions,
                ground_obs_next,
            )
        else:
            ground_obs = model.env.env_method("to_ground_obs", model._last_obs[0])
            ground_obs_next = model.env.env_method(
                "to_ground_obs", obs_next_terminal[0]
            )
            result = model.env.env_method(
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

        action_dim = self._counterfactual_action_dim()
        obs_shape = model.env.observation_space.shape
        assert obs_shape is not None
        c_obs = self.reshape_and_trim(
            c_obs,
            final_dim=obs_shape[0],
        )
        c_actions = self.reshape_and_trim(c_actions, final_dim=action_dim)
        c_obs_next = self.reshape_and_trim(
            c_obs_next,
            final_dim=obs_shape[0],
        )
        c_rewards = self.reshape_and_trim(c_rewards, final_dim=1)
        c_dones = self.reshape_and_trim(c_dones, final_dim=1)
        c_infos = self.reshape_and_trim(c_infos, final_dim=1)

        for i in range(len(c_obs)):
            replay_buffer.add(
                obs=c_obs[i],
                next_obs=c_obs_next[i],
                action=c_actions[i],
                reward=c_rewards[i],
                done=c_dones[i],
                infos=c_infos[i],
            )

        model._last_obs = obs_next

    def _counterfactual_action_dim(self) -> int:
        model = cast(Any, self)
        action_shape = getattr(model.env.action_space, "shape", ())
        if len(action_shape) == 0:
            return 1
        return int(action_shape[0])

    def reshape_and_trim(self, array: np.ndarray, final_dim: int) -> np.ndarray:
        """Trim into batches to match the number of vectorized environments."""
        model = cast(Any, self)
        assert isinstance(model.env, VecEnv), "You must pass a VecEnv"

        if final_dim > 1:
            target_shape = (-1, model.env.num_envs, final_dim)
        else:
            target_shape = (-1, model.env.num_envs)
        num_elements_per_batch = model.env.num_envs * final_dim

        flat_array = array.flatten()
        num_elements_required = num_elements_per_batch * (
            len(flat_array) // num_elements_per_batch
        )
        trimmed_array = flat_array[:num_elements_required]
        return trimmed_array.reshape(target_shape)
