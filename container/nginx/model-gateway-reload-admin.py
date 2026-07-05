from __future__ import annotations

import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="Model Gateway Reload Admin")
_reload_lock = threading.Lock()
_allowed_canary_percents = {0, 1, 5, 10, 50, 100}
_traffic_policy_dir = Path(os.environ.get("NGINX_TRAFFIC_POLICY_DIR", "/etc/nginx/traffic-policies"))
_runtime_conf = Path(os.environ.get("NGINX_RUNTIME_CONF", "/etc/nginx/runtime/model-production-upstream.conf"))


class ReloadRequest(BaseModel):
    request_id: str | None = None
    canary_percent: int | None = Field(default=None, ge=0, le=100)


class TrafficPolicyRequest(BaseModel):
    request_id: str | None = None
    canary_percent: int = Field(ge=0, le=100)
    dry_run: bool = False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/admin/traffic-policy")
def get_traffic_policy() -> dict[str, Any]:
    return _traffic_policy_state()


@app.post("/admin/traffic-policy")
def set_traffic_policy(payload: TrafficPolicyRequest) -> dict[str, Any]:
    if payload.canary_percent not in _allowed_canary_percents:
        raise HTTPException(
            status_code=422,
            detail={
                "status": "failed",
                "request_id": payload.request_id,
                "reason": "unsupported_canary_percent",
                "canary_percent": payload.canary_percent,
                "allowed_canary_percents": sorted(_allowed_canary_percents),
            },
        )

    policy_path = _traffic_policy_dir / f"canary-{payload.canary_percent:03d}.conf"
    if not policy_path.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "status": "failed",
                "request_id": payload.request_id,
                "reason": "traffic_policy_not_found",
                "canary_percent": payload.canary_percent,
                "policy_path": str(policy_path),
            },
        )

    policy_text = policy_path.read_text(encoding="utf-8")
    if payload.dry_run:
        return {
            "status": "dry_run",
            "request_id": payload.request_id,
            "canary_percent": payload.canary_percent,
            "policy_path": str(policy_path),
            "active_conf": str(_runtime_conf),
            "policy": policy_text,
        }

    if not _reload_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="nginx reload already in progress",
        )

    previous_conf: str | None = None
    had_previous_conf = _runtime_conf.exists()

    try:
        if had_previous_conf:
            previous_conf = _runtime_conf.read_text(encoding="utf-8")

        _runtime_conf.parent.mkdir(parents=True, exist_ok=True)
        tmp_conf = _runtime_conf.with_name(f"{_runtime_conf.name}.tmp")
        tmp_conf.write_text(policy_text, encoding="utf-8")
        tmp_conf.replace(_runtime_conf)

        test_result = _run_command(["nginx", "-t"])
        if test_result["returncode"] != 0:
            rollback_result = _restore_runtime_conf(previous_conf, had_previous_conf)
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "failed",
                    "request_id": payload.request_id,
                    "reason": "nginx_config_test_failed",
                    "canary_percent": payload.canary_percent,
                    "policy_path": str(policy_path),
                    "nginx_test": test_result,
                    "rollback": rollback_result,
                },
            )

        reload_result = _run_command(["nginx", "-s", "reload"])
        if reload_result["returncode"] != 0:
            rollback_result = _restore_runtime_conf(previous_conf, had_previous_conf)
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "failed",
                    "request_id": payload.request_id,
                    "reason": "nginx_reload_failed",
                    "canary_percent": payload.canary_percent,
                    "policy_path": str(policy_path),
                    "nginx_test": test_result,
                    "nginx_reload": reload_result,
                    "rollback": rollback_result,
                },
            )

        result = {
            "status": "succeeded",
            "request_id": payload.request_id,
            "canary_percent": payload.canary_percent,
            "policy_path": str(policy_path),
            "active_conf": str(_runtime_conf),
            "reloaded_at_epoch": time.time(),
            "nginx_test": test_result,
            "nginx_reload": reload_result,
        }

        print(
            "model_gateway_traffic_policy_applied "
            f"request_id={payload.request_id} "
            f"canary_percent={payload.canary_percent} "
            f"policy_path={policy_path}",
            flush=True,
        )

        return result
    finally:
        _reload_lock.release()


@app.post("/admin/reload")
def reload_nginx(payload: ReloadRequest) -> dict[str, Any]:
    if not _reload_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="nginx reload already in progress",
        )

    try:
        test_result = _run_command(["nginx", "-t"])
        if test_result["returncode"] != 0:
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "failed",
                    "request_id": payload.request_id,
                    "reason": "nginx_config_test_failed",
                    "nginx_test": test_result,
                },
            )

        reload_result = _run_command(["nginx", "-s", "reload"])
        if reload_result["returncode"] != 0:
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "failed",
                    "request_id": payload.request_id,
                    "reason": "nginx_reload_failed",
                    "nginx_test": test_result,
                    "nginx_reload": reload_result,
                },
            )

        result = {
            "status": "succeeded",
            "request_id": payload.request_id,
            "canary_percent": payload.canary_percent,
            "reloaded_at_epoch": time.time(),
            "nginx_test": test_result,
            "nginx_reload": reload_result,
        }

        print(
            "model_gateway_nginx_reloaded "
            f"request_id={payload.request_id} "
            f"canary_percent={payload.canary_percent}",
            flush=True,
        )

        return result
    finally:
        _reload_lock.release()


def _restore_runtime_conf(previous_conf: str | None, had_previous_conf: bool) -> dict[str, Any]:
    if had_previous_conf:
        _runtime_conf.write_text(previous_conf or "", encoding="utf-8")
        rollback_test = _run_command(["nginx", "-t"])
        return {
            "status": "restored_previous_conf",
            "active_conf": str(_runtime_conf),
            "nginx_test": rollback_test,
        }

    try:
        _runtime_conf.unlink()
    except FileNotFoundError:
        pass

    return {
        "status": "removed_new_conf",
        "active_conf": str(_runtime_conf),
    }


def _traffic_policy_state() -> dict[str, Any]:
    allowed_canary_percents = sorted(_allowed_canary_percents)
    if not _runtime_conf.exists():
        return {
            "status": "not_initialized",
            "active_conf": str(_runtime_conf),
            "active_conf_exists": False,
            "canary_percent": None,
            "matched_policy_path": None,
            "allowed_canary_percents": allowed_canary_percents,
            "policy": None,
        }

    policy_text = _runtime_conf.read_text(encoding="utf-8")
    matched_policy = _match_traffic_policy(policy_text)
    canary_percent = matched_policy[0] if matched_policy else None
    matched_policy_path = str(matched_policy[1]) if matched_policy else None

    return {
        "status": "matched" if matched_policy else "custom",
        "active_conf": str(_runtime_conf),
        "active_conf_exists": True,
        "canary_percent": canary_percent,
        "matched_policy_path": matched_policy_path,
        "allowed_canary_percents": allowed_canary_percents,
        "policy": policy_text,
    }


def _match_traffic_policy(policy_text: str) -> tuple[int, Path] | None:
    normalized_policy = policy_text.strip()
    for canary_percent in sorted(_allowed_canary_percents):
        policy_path = _traffic_policy_dir / f"canary-{canary_percent:03d}.conf"
        if not policy_path.is_file():
            continue

        candidate_text = policy_path.read_text(encoding="utf-8").strip()
        if normalized_policy == candidate_text:
            return canary_percent, policy_path

    return None


def _run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
