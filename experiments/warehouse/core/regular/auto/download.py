import os
import subprocess
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

load_dotenv()


def download_tensorboard_logs(
    hostname: str,
    username: str,
    remote_dir: str,
    local_dir: str,
) -> None:
    """Download TensorBoard logs using rsync.

    Args:
        hostname: Remote hostname
        username: SSH username
        remote_dir: Remote logs directory
        local_dir: Local directory to save logs
    """
    # Create local directory
    os.makedirs(local_dir, exist_ok=True)

    # Construct rsync command
    remote_path = f"{username}@{hostname}:{remote_dir}/"
    rsync_command = [
        "rsync",
        "-avz",  # archive mode, verbose, compress
        "--progress",  # show progress
        remote_path,
        local_dir,
    ]

    # Execute rsync command
    subprocess.run(rsync_command, check=True)


def download_videos(
    hostname: str,
    username: str,
    remote_dir: str,
    local_dir: str,
) -> None:
    """Download videos using rsync.

    Args:
        hostname: Remote hostname
        username: SSH username
        remote_dir: Remote videos directory
        local_dir: Local directory to save videos
    """
    # Create local directory
    os.makedirs(local_dir, exist_ok=True)

    # Construct rsync command to download only .mp4 files
    remote_path = f"{username}@{hostname}:{remote_dir}/"
    rsync_command = [
        "rsync",
        "-avz",  # archive mode, verbose, compress
        "--include=*.mp4",  # include only .mp4 files
        "--exclude=*.json",  # exclude everything else
        remote_path,
        local_dir,
    ]

    # Execute rsync command
    subprocess.run(rsync_command, check=True)


@hydra.main(
    config_path="../../core/conf",
    config_name="config",
    version_base=None,
)
def main(config: DictConfig) -> None:
    """Download TensorBoard logs from remote server.

    Args:
        config: Hydra configuration
    """
    # Create local download directory
    local_log_dir = Path("logs")
    local_log_dir.mkdir(exist_ok=True)

    local_video_dir = Path("videos")
    local_video_dir.mkdir(exist_ok=True)

    # Download logs directory
    download_tensorboard_logs(
        hostname=os.environ["CLUSTER_HOST"],
        username=os.environ["CLUSTER_USER"],
        remote_dir=config.logs.remote_dir,
        local_dir=str(local_log_dir),
    )

    # Download videos directory
    download_videos(
        hostname=os.environ["CLUSTER_HOST"],
        username=os.environ["CLUSTER_USER"],
        remote_dir=config.videos.remote_dir,
        local_dir=str(local_video_dir),
    )


if __name__ == "__main__":
    raise SystemExit(main())
