#!/usr/bin/env python3
"""Vast.ai instance automation: start -> run SSH job -> collect logs -> stop.

Usage example:
  export VAST_API_KEY="..."
  python3 scripts/vast_instance_cycle.py \
    --instance-id 12345678 \
    --remote-repo-dir /root/flashinfer-bench-starter-kit \
    --remote-command "python3 scripts/run_all_workloads_short.sh --max-workloads 19" \
    --ssh-key ~/.ssh/id_rsa
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


API_BASE = "https://console.vast.ai/api/v0"


class RateLimitError(RuntimeError):
    """Raised when Vast API returns HTTP 429."""


def _api_request(api_key: str, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{API_BASE}{path}"
    data = None
    headers = {"Authorization": f"Bearer {api_key}"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        if e.code == 429:
            raise RateLimitError(f"{method} {path} failed: HTTP 429 {body}") from e
        raise RuntimeError(f"{method} {path} failed: HTTP {e.code} {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"{method} {path} failed: {e}") from e


def _set_instance_state(api_key: str, instance_id: int, state: str) -> dict[str, Any]:
    return _api_request(api_key, "PUT", f"/instances/{instance_id}/", {"state": state})


def _show_instance(api_key: str, instance_id: int) -> dict[str, Any]:
    return _api_request(api_key, "GET", f"/instances/{instance_id}/")


def _request_instance_logs(api_key: str, instance_id: int, tail: int) -> dict[str, Any]:
    return _api_request(
        api_key,
        "PUT",
        f"/instances/request_logs/{instance_id}",
        {"tail": str(tail), "daemon_logs": "false"},
    )


def _download_text(url: str) -> str:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _wait_until_running(
    api_key: str,
    instance_id: int,
    timeout_sec: int,
    poll_sec: float,
) -> tuple[str, int]:
    deadline = time.time() + timeout_sec
    last_state = "unknown"
    backoff_sec = max(15.0, poll_sec * 2.0)
    while time.time() < deadline:
        try:
            payload = _show_instance(api_key, instance_id)
        except RateLimitError:
            time.sleep(backoff_sec)
            continue
        inst = payload.get("instances", {})
        state = str(inst.get("actual_status") or inst.get("cur_state") or "unknown").lower()
        last_state = state
        host = inst.get("ssh_host")
        port = inst.get("ssh_port")
        if state == "running" and host and port:
            return str(host), int(port)
        if state in {"destroyed", "terminated"}:
            raise RuntimeError(f"Instance {instance_id} entered terminal state: {state}")
        time.sleep(poll_sec)
    raise TimeoutError(f"Timed out waiting for running state (last state: {last_state})")


def _build_remote_script(
    repo_dir: str,
    git_pull: bool,
    remote_command: str,
    remote_timeout_sec: int,
) -> str:
    parts = ["set -euo pipefail", f"cd {shlex.quote(repo_dir)}"]
    # Best-effort remote conda bootstrap for benchmark runs.
    parts.extend(
        [
            "if [ -f /workspace/miniconda3/etc/profile.d/conda.sh ]; then source /workspace/miniconda3/etc/profile.d/conda.sh; "
            "elif [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then source /root/miniconda3/etc/profile.d/conda.sh; "
            "elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then source /opt/conda/etc/profile.d/conda.sh; fi",
            "if command -v conda >/dev/null 2>&1; then conda activate fi-bench || true; fi",
        ]
    )
    if git_pull:
        parts.append("git pull --ff-only")
    if remote_timeout_sec > 0 and "timeout " not in remote_command:
        wrapped = f"timeout -k 5s {int(remote_timeout_sec)}s bash -lc {shlex.quote(remote_command)}"
        parts.append(wrapped)
    else:
        parts.append(remote_command)
    return "\n".join(parts)


def _run_ssh(
    ssh_user: str,
    ssh_host: str,
    ssh_port: int,
    ssh_key: str | None,
    remote_script: str,
    ssh_connect_timeout: int,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=10",
        "-o",
        f"ConnectTimeout={ssh_connect_timeout}",
        "-p",
        str(ssh_port),
    ]
    if ssh_key:
        cmd.extend(["-i", ssh_key])
    cmd.append(f"{ssh_user}@{ssh_host}")
    cmd.extend(["bash", "-lc", remote_script])
    return subprocess.run(cmd, text=True, capture_output=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Automate Vast.ai instance execution via SSH")
    p.add_argument("--instance-id", type=int, required=True, help="Vast instance(contract) ID")
    p.add_argument("--remote-repo-dir", type=str, default="/root/flashinfer-bench-starter-kit")
    p.add_argument("--remote-command", type=str, required=True, help="Command to run on remote host")
    p.add_argument("--ssh-user", type=str, default="root")
    p.add_argument("--ssh-key", type=str, default=None, help="Path to local SSH private key")
    p.add_argument("--no-git-pull", action="store_true", help="Skip `git pull --ff-only` before command")
    p.add_argument("--keep-running", action="store_true", help="Do not stop instance at the end")
    p.add_argument("--start-timeout-sec", type=int, default=600)
    p.add_argument("--poll-sec", type=float, default=5.0)
    p.add_argument("--ssh-connect-timeout-sec", type=int, default=30)
    p.add_argument(
        "--remote-timeout-sec",
        type=int,
        default=30,
        help="Wrap remote command with GNU timeout; set 0 to disable wrapper",
    )
    p.add_argument("--vast-log-tail", type=int, default=4000)
    p.add_argument("--log-dir", type=str, default="logs/vast")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("VAST_API_KEY")
    if not api_key:
        print("ERROR: VAST_API_KEY environment variable is required.", file=sys.stderr)
        return 2

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ssh_log_path = log_dir / f"vast_{args.instance_id}_{ts}_ssh.log"
    vast_log_path = log_dir / f"vast_{args.instance_id}_{ts}_container.log"
    meta_path = log_dir / f"vast_{args.instance_id}_{ts}_meta.json"

    meta: dict[str, Any] = {
        "instance_id": args.instance_id,
        "start_time": dt.datetime.now().isoformat(),
        "remote_repo_dir": args.remote_repo_dir,
        "remote_command": args.remote_command,
    }

    started_here = False
    try:
        print(f"[vast] start instance {args.instance_id}")
        _set_instance_state(api_key, args.instance_id, "running")
        started_here = True

        print(f"[vast] waiting for running state (timeout={args.start_timeout_sec}s)")
        ssh_host, ssh_port = _wait_until_running(
            api_key=api_key,
            instance_id=args.instance_id,
            timeout_sec=args.start_timeout_sec,
            poll_sec=args.poll_sec,
        )
        meta["ssh_host"] = ssh_host
        meta["ssh_port"] = ssh_port
        print(f"[vast] running: {ssh_host}:{ssh_port}")

        remote_script = _build_remote_script(
            repo_dir=args.remote_repo_dir,
            git_pull=not args.no_git_pull,
            remote_command=args.remote_command,
            remote_timeout_sec=args.remote_timeout_sec,
        )
        print("[vast] executing remote command over ssh")
        proc = _run_ssh(
            ssh_user=args.ssh_user,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            ssh_key=args.ssh_key,
            remote_script=remote_script,
            ssh_connect_timeout=args.ssh_connect_timeout_sec,
        )

        ssh_log_path.write_text(
            "=== STDOUT ===\n"
            + proc.stdout
            + "\n=== STDERR ===\n"
            + proc.stderr
            + f"\n=== EXIT_CODE ===\n{proc.returncode}\n",
            encoding="utf-8",
        )
        meta["ssh_exit_code"] = proc.returncode
        print(f"[vast] ssh log saved: {ssh_log_path}")

        try:
            print("[vast] requesting container logs")
            lr = _request_instance_logs(api_key, args.instance_id, args.vast_log_tail)
            log_url = lr.get("result_url")
            if log_url:
                text = _download_text(str(log_url))
                vast_log_path.write_text(text, encoding="utf-8")
                meta["vast_result_url"] = log_url
                print(f"[vast] vast logs saved: {vast_log_path}")
            else:
                meta["vast_log_request"] = lr
                print(f"[vast] no result_url in log response: {lr}")
        except Exception as e:  # noqa: BLE001
            meta["vast_log_error"] = str(e)
            print(f"[vast] log fetch failed: {e}", file=sys.stderr)

        if proc.returncode != 0:
            print(f"[vast] remote command failed with exit code {proc.returncode}", file=sys.stderr)
            return proc.returncode
        return 0
    finally:
        meta["end_time"] = dt.datetime.now().isoformat()
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[vast] meta saved: {meta_path}")
        if started_here and not args.keep_running:
            try:
                print(f"[vast] stop instance {args.instance_id}")
                _set_instance_state(api_key, args.instance_id, "stopped")
            except Exception as e:  # noqa: BLE001
                print(f"[vast] failed to stop instance: {e}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
