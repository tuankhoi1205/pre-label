"""Deploy the local GPU function via Nuclio's documented dashboard API.

The Windows nuctl local platform requires /bin/sh, so use the dashboard, which
already runs on Linux with the upstream Docker socket and workspace mounts.
This registers the function; it never invokes the model or annotates a task.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

import yaml


def request(base, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
        headers={'Content-Type': 'application/json',
                 'x-nuclio-function-namespace': 'nuclio',
                 'x-nuclio-project-namespace': 'nuclio'})
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            raw = response.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        if method == 'GET' and exc.code == 404:
            return None
        raise RuntimeError(f'Nuclio {method} {path}: HTTP {exc.code}: '
                           f'{exc.read().decode(errors="replace")}') from exc


def deploy(base, config):
    function = yaml.safe_load(Path(config).read_text(encoding='utf-8'))
    name = function['metadata']['name']
    if name != 'pth-prelabel-centerpoint':
        raise ValueError('This deployment helper is limited to the CenterPoint function')
    if request(base, 'GET', '/api/projects/cvat') is None:
        request(base, 'POST', '/api/projects', {
            'metadata': {'name': 'cvat', 'namespace': 'nuclio'},
            'spec': {'description': 'Local CVAT automatic annotation'}})
    function['metadata'].setdefault('labels', {})['nuclio.io/project-name'] = 'cvat'
    function['spec']['build']['path'] = '/workspace/deployment/nuclio'
    existing = request(base, 'GET', f'/api/functions/{name}')
    if existing:
        state = existing.get('status', {}).get('state')
        if state in ('building', 'deploying', 'waitingForResourceConfiguration'):
            print(f'CenterPoint deployment already in progress ({state})', flush=True)
            return name
        # PUT requires a complete spec, including the server's image/version.
        spec = dict(existing['spec'])
        spec.update(function['spec'])
        function['spec'] = spec
        request(base, 'PUT', f'/api/functions/{name}', function)
    else:
        request(base, 'POST', '/api/functions', function)
    print(f'CenterPoint deployment accepted: {name}', flush=True)
    return name


def wait_ready(base, name, timeout):
    deadline, previous = time.monotonic() + timeout, None
    while time.monotonic() < deadline:
        result = request(base, 'GET', f'/api/functions/{name}')
        status = (result or {}).get('status', {})
        state = status.get('state', 'not_found')
        if state != previous:
            print(f'CenterPoint status: {state}', flush=True)
            previous = state
        if state == 'ready':
            return result
        if state in ('error', 'unhealthy'):
            raise RuntimeError(json.dumps(status, ensure_ascii=False))
        time.sleep(5)
    raise TimeoutError('CenterPoint deployment did not become ready; inspect Nuclio logs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://localhost:8070')
    parser.add_argument('--config', default='.local/function-gpu.generated.yaml')
    parser.add_argument('--wait-only', action='store_true')
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    name = 'pth-prelabel-centerpoint' if args.wait_only else deploy(args.url, args.config)
    result = wait_ready(args.url, name, args.timeout)
    output = Path('.local/centerpoint-function-status.json')
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Ready. Status saved to {output}. No inference has been run.', flush=True)
