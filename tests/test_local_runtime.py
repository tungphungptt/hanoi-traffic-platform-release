"""Validate startup using effective child-process flags, not just a healthy port."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

import pytest
from fastapi.testclient import TestClient
import yaml

import app.main as main
from tools.check_local_runtime import validate_local_status

ROOT = Path(__file__).resolve().parents[1]


def status_payloads():
    return ({'ok': True}, {'process_id': 123, 'public_demo_mode': False},
            {'public_demo_mode': False, 'video_enabled': True, 'enabled': True,
             'artifacts_present': True, 'missing_files': [], 'video_review_enabled': True, 'hanoi_forecaster': {'artifacts_present': True}})


def test_health_alone_is_not_local_readiness():
    health, system, aicc = status_payloads()
    validate_local_status(health, system, aicc, 123)


@pytest.mark.parametrize('part,key,value', [
    (0, 'ok', False), (1, 'process_id', 999), (1, 'public_demo_mode', True),
    (2, 'public_demo_mode', True), (2, 'video_enabled', False), (2, 'enabled', False),
    (2, 'video_review_enabled', False), (2, 'hanoi_forecaster', {'artifacts_present': False}),
    (2, 'artifacts_present', False), (2, 'missing_files', ['detector.onnx']),
])
def test_unready_or_wrong_child_is_rejected(part, key, value):
    payloads = status_payloads()
    payloads[part][key] = value
    with pytest.raises(ValueError):
        validate_local_status(*payloads, 123)


@pytest.mark.parametrize('public,video', [(True, True), (False, False), (False, True)])
def test_effective_flags_have_one_backend_source(monkeypatch, public, video):
    monkeypatch.setattr(main, 'PUBLIC_VIDEO_REVIEW_ENABLED', False)
    monkeypatch.setattr(main, 'PUBLIC_DEMO_MODE', public)
    monkeypatch.setattr(main, 'AICC_VIDEO_ENABLED', video)
    client = TestClient(main.app)
    status = client.get('/api/aicc/status').json()
    system = client.get('/api/system').json()
    assert system['video_review'] == {key: status[key] for key in system['video_review']}
    assert status['enabled'] is (video and not public)
    assert status['public_demo_mode'] is public
    assert status['video_enabled'] is video
    assert system['process_id'] == os.getpid()
    if public or not video:
        assert client.post('/api/scenarios/video-review-from-data', json={'path': 'clip.mp4'}).status_code == 403
        assert client.post('/api/scenarios/video-review', files={'file': ('clip.mp4', b'', 'video/mp4')}).status_code == 403


def test_public_deployment_enables_only_video_review():
    config = yaml.safe_load((ROOT / 'render.yaml').read_text())
    values = {row['key']: row['value'] for row in config['services'][0]['envVars']}
    assert values['PUBLIC_DEMO_MODE'] == 'true'
    assert values['AICC_VIDEO_ENABLED'] == 'false'
    assert values['PUBLIC_VIDEO_REVIEW_ENABLED'] == 'true'


def test_process_environment_wins_over_dotenv_in_child(tmp_path, monkeypatch):
    dotenv = tmp_path / '.env'
    dotenv.write_text('PUBLIC_DEMO_MODE=true\nAICC_VIDEO_ENABLED=false\n')
    monkeypatch.setenv('PUBLIC_DEMO_MODE', 'false')
    monkeypatch.setenv('AICC_VIDEO_ENABLED', 'true')
    code = '''from pathlib import Path
import json, sys
from traffic_brain.providers import load_dotenv
load_dotenv(Path(sys.argv[1]))
from app.main import video_review_access
print(json.dumps(video_review_access()))
'''
    child = subprocess.run([sys.executable, '-c', code, str(dotenv)], cwd=ROOT,
                           capture_output=True, text=True, check=True, timeout=20)
    assert json.loads(child.stdout) == {'public_demo_mode': False, 'video_enabled': True,
                                       'enabled': True, 'blocked_reason': None}


@pytest.mark.parametrize('public', [False, True])
def test_real_child_http_mode_and_startup_checker(public):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    child_env = os.environ.copy()
    child_env.update(PUBLIC_DEMO_MODE=str(public).lower(), AICC_VIDEO_ENABLED='true', PUBLIC_VIDEO_REVIEW_ENABLED='false', FORECAST_PROVIDER='local_onnx', HANOI_LAUNCH_INSTANCE='test-child-instance')
    child = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', str(port)],
                             cwd=ROOT, env=child_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f'http://127.0.0.1:{port}'
    opener = build_opener(ProxyHandler({}))
    try:
        deadline = time.monotonic() + 20
        while True:
            assert child.poll() is None, f'Child server exited: {child.returncode}'
            try:
                with opener.open(base + '/api/system', timeout=0.5) as response:
                    system = json.load(response)
                break
            except (URLError, OSError):
                assert time.monotonic() < deadline, 'Child server startup timed out'
                time.sleep(0.05)
        assert system['process_id'] == child.pid
        assert system['public_demo_mode'] is public
        assert system['video_review']['enabled'] is (not public)
        check = subprocess.run([sys.executable, 'tools/check_local_runtime.py', '--base-url', base,
                                '--expected-instance', 'test-child-instance'], cwd=ROOT, capture_output=True, text=True, timeout=15)
        assert check.returncode == (2 if public else 0), check.stdout + check.stderr
        request = Request(base + '/api/scenarios/video-review-from-data', data=b'{"path":"missing-test.mp4"}',
                          headers={'Content-Type': 'application/json'})
        with pytest.raises(HTTPError) as error:
            opener.open(request, timeout=3)
        assert error.value.code == (403 if public else 404)
    finally:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


def test_windows_redirector_pid_does_not_break_matching_launch_instance():
    health, system, aicc = status_payloads()
    system['launch_instance'] = 'new-local-instance'
    validate_local_status(health, system, aicc, expected_instance='new-local-instance')
    with pytest.raises(ValueError, match='launch instance'):
        validate_local_status(health, system, aicc, expected_instance='stale-instance')
