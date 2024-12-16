import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import paramiko
from paramiko import SSHClient


def parse_ls_line(line: str) -> Optional[Tuple[str, datetime]]:
    """Parse a single line of ls output to extract filename and timestamp."""
    pattern = r".*?\s+(\w+)\s+(\d+)\s+(\d+):(\d+)\s+(model_\d+_steps\.zip)"
    match = re.search(pattern, line)

    if not match:
        return None

    month, day, hour, minute, filename = match.groups()
    current_year = datetime.now().year
    month_num = datetime.strptime(month, "%b").month

    timestamp = datetime(current_year, month_num, int(day), int(hour), int(minute))
    return filename, timestamp


def get_run_names(ls_output: str) -> list[str]:
    """Get the names of all runs from the ls output."""
    run_names = []

    for line in ls_output.splitlines()[1:]:
        run_name = line.split()[-1]
        run_names.append(run_name)
    return run_names


def get_latest_checkpoint(ls_output: str) -> str:
    """Find the most recent checkpoint file from ls output."""
    checkpoints = []

    for line in ls_output.splitlines():
        result = parse_ls_line(line)
        if result:
            checkpoints.append(result)

    if not checkpoints:
        raise ValueError("No checkpoint files found in ls output")

    latest_file, _ = max(checkpoints, key=lambda x: x[1])
    return latest_file


def download_latest_checkpoint(
    remote_host: str,
    remote_path: str,
    local_path: Path,
    username: Optional[str] = None,
    key_filename: Optional[str] = None,
) -> None:
    """Download the latest checkpoint from the remote server using Paramiko.

    Args:
        remote_host: Hostname of the remote server
        remote_path: Path to checkpoint directory on remote server
        local_path: Local directory to save the checkpoint
        username: SSH username
        key_filename: Path to SSH private key file

    Returns:
        Path to the downloaded checkpoint file
    """
    # Set up SSH client
    ssh = SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(key_filename)
        # Connect to remote server
        ssh.connect(remote_host, username=username, key_filename=key_filename)

        # Get directory listing
        stdin, stdout, stderr = ssh.exec_command(f"ls -l {remote_path}")
        ls_output = stdout.read().decode()

        if stderr.channel.recv_exit_status() != 0:
            raise RuntimeError(
                f"Failed to get directory listing: {stderr.read().decode()}"
            )

        run_names = get_run_names(ls_output)

        for run_name in run_names:
            print(run_name)
            stdin, stdout, stderr = ssh.exec_command(f"ls -l {remote_path}/{run_name}")
            ls_output = stdout.read().decode()

            if stderr.channel.recv_exit_status() != 0:
                raise RuntimeError(
                    f"Failed to get directory listing: {stderr.read().decode()}"
                )

            latest_file = get_latest_checkpoint(ls_output)

            local_file = local_path / run_name / latest_file

            if local_file.exists():
                print(f"Checkpoint already exists: {local_file}")
                continue

            print(f"Downloading checkpoint: {local_file}")

            # Create local directory
            local_file.parent.mkdir(parents=True, exist_ok=True)

            # Download using SFTP
            with ssh.open_sftp() as sftp:
                remote_file_path = f"{remote_path}/{run_name}/{latest_file}"
                sftp.get(remote_file_path, str(local_file))
    finally:
        ssh.close()


if __name__ == "__main__":
    # Example usage
    REMOTE_HOST = os.environ["CLUSTER_HOST"]
    REMOTE_PATH = "/datasets/tbester/warehouse/checkpoints/"
    LOCAL_PATH = Path("./downloaded_checkpoints")
    USERNAME = os.environ["CLUSTER_USER"]
    KEY_FILE = os.path.expanduser("~/.ssh/id_rsa")  # This will expand to the full path

    download_latest_checkpoint(REMOTE_HOST, REMOTE_PATH, LOCAL_PATH, USERNAME, KEY_FILE)
