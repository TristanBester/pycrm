from typing import Any

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices

# class DispatchSubprocVecEnv(SubprocVecEnv):
#     """Enables user to dispatch method calls to multiple environments in parallel."""

#     def dispatched_env_method(
#         self,
#         method_name: str,
#         *method_args,
#         indices: VecEnvIndices | None = None,
#     ) -> list[Any]:
#         """Dispatch a method call to the specified environments.

#         Args:
#             method_name: The name of the method to call.
#             *method_args: The arguments to pass to the method.
#             indices: The indices of the environments to call the method on.

#         Returns:
#             A list of the return values of the method calls.
#         """
#         target_remotes = self._get_target_remotes(indices)

#         for job in zip(target_remotes, *method_args, strict=True):
#             remote, args = job[0], job[1:]
#             remote.send(("env_method", (method_name, args, {})))
#         return [remote.recv() for remote in target_remotes]


class DispatchSubprocVecEnv(SubprocVecEnv):
    def dispatched_env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
    ):
        """Dispatches a method call to the specified environments.

        Args:
            method_name (str): The method to call.
            *method_args: The arguments to pass to the method.
            indices (VecEnvIndices): The indices of the environments to call the method on.
            **method_kwargs: The keyword arguments to pass to the method.

        Returns:
            A list of results for the method call of each environment.

        """
        target_remotes = self._get_target_remotes(indices)
        assert (
            len(target_remotes) == method_args[0].shape[0]
        ), "Number of remotes and method_args should be the same"

        for job in zip(target_remotes, *method_args):
            remote, args = job[0], job[1:]
            remote.send(("env_method", (method_name, args, {})))
        return [remote.recv() for remote in target_remotes]
