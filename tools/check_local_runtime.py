"""Verify the local child process and effective video permissions before opening its UI."""
from __future__ import annotations

import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener


def validate_local_status(health: dict, system: dict, aicc: dict, expected_pid: int | None = None, expected_instance: str | None = None) -> None:
    if health.get("ok") is not True:
        raise ValueError("Backend is not healthy.")
    if system.get("public_demo_mode") is not False or aicc.get("public_demo_mode") is not False:
        raise ValueError("Backend is still read-only; set PUBLIC_DEMO_MODE=false in the Uvicorn child environment.")
    if aicc.get("video_enabled") is not True or aicc.get("enabled") is not True:
        raise ValueError("Video is disabled; set AICC_VIDEO_ENABLED=true in the Uvicorn child environment.")
    actual_pid = system.get("process_id")
    if not isinstance(actual_pid, int) or isinstance(actual_pid, bool) or actual_pid <= 0:
        raise ValueError("Backend does not report a valid server PID; update the local checkout.")
    if expected_instance is not None and system.get("launch_instance") != expected_instance:
        raise ValueError("Wrong launch instance or stale code at this port; stop the old backend and check the checkout.")
    if expected_pid is not None and system.get("process_id") != expected_pid:
        raise ValueError("Wrong process or stale code at this port; stop the old backend and check the checkout.")
    if aicc.get("artifacts_present") is not True or aicc.get("missing_files"):
        raise ValueError("AICC model bundle is missing; check models/ in this local checkout.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--expected-pid", type=int)
    identity.add_argument("--expected-instance")
    args = parser.parse_args()
    url = urlsplit(args.base_url)
    if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or url.username or url.password or url.path not in {"", "/"} or url.query or url.fragment):
        parser.error("--base-url must be a loopback HTTP origin")
    # Local health checks must reach the child directly, even if the shell has HTTP_PROXY.
    opener = build_opener(ProxyHandler({}))
    payloads = []
    for path in ("/api/health", "/api/system", "/api/aicc/status"):
        try:
            with opener.open(args.base_url.rstrip("/") + path, timeout=2) as response:
                payload = json.loads(response.read(1024 * 1024))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object")
                payloads.append(payload)
        except HTTPError as exc:
            print(f"LOCAL_RUNTIME_WAIT: {path} HTTP {exc.code}")
            return 1
        except (URLError, OSError, ValueError):
            print(f"LOCAL_RUNTIME_WAIT: cannot read JSON from {path}")
            return 1
    try:
        validate_local_status(*payloads, expected_pid=args.expected_pid, expected_instance=args.expected_instance)
    except ValueError as exc:
        print(f"LOCAL_RUNTIME_BLOCKED: {exc}")
        return 2
    actual_pid = payloads[1]["process_id"]
    print(f"LOCAL_RUNTIME_READY: pid={actual_pid}, PUBLIC_DEMO_MODE=false, AICC_VIDEO_ENABLED=true, artifacts_present=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
