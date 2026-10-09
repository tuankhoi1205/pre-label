"""Fetch/verify the official Linux Python 3.10 CUDA wheels in a resumable cache."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TORCH_HASH = '14c5c9db09df8cf1b3942a3479c779da6e293a84a162d8a6ac71e2bde30e30c5'
WHEELS = {
    'torch': ('torch-1.13.1+cu117-cp310-cp310-linux_x86_64.whl', 'https://download.pytorch.org/whl/cu117/torch/'),
    'torchvision': ('torchvision-0.14.1+cu117-cp310-cp310-linux_x86_64.whl', 'https://download.pytorch.org/whl/cu117/torchvision/'),
    'mmcv': ('mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl', 'https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html'),
}


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.extend(value for key, value in attrs if key == 'href')


def sources():
    result = {}
    for package, (filename, index) in WHEELS.items():
        parser = Links()
        parser.feed(urllib.request.urlopen(index, timeout=120).read().decode())
        matches = [urllib.parse.urljoin(index, link) for link in parser.links
                   if urllib.parse.unquote(urllib.parse.urlsplit(link).path).endswith('/' + filename)]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one official wheel for {filename}')
        linked = urllib.parse.urlsplit(matches[0])
        expected = urllib.parse.parse_qs(linked.fragment).get('sha256', [None])[0]
        if package == 'torch' and expected != TORCH_HASH:
            raise ValueError('Official Torch hash differs from the audited runtime wheel')
        if package in ('torch', 'torchvision') and not expected:
            raise ValueError('Official PyTorch wheel lacks its SHA256')
        # The original official endpoint avoids an incomplete R2 transfer observed here.
        url = urllib.parse.urlunsplit(linked._replace(netloc='download.pytorch.org', fragment='')) if package != 'mmcv' else urllib.parse.urlunsplit(linked._replace(fragment=''))
        result[package] = {'filename': filename, 'url': url, 'index': index, 'expected_sha256': expected}
    return result


def verify(directory):
    doc = json.loads((directory / 'runtime-wheels.json').read_text())
    if doc['torch']['sha256'] != TORCH_HASH:
        raise ValueError('Torch checksum is not the audited official checksum')
    for package, (filename, _) in WHEELS.items():
        path = directory / filename
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else checksum(stream)
        if actual != doc[package]['sha256']:
            raise ValueError(f'Cached wheel checksum mismatch: {package}')
    print('Official runtime wheel cache verified')


def checksum(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def fetch():
    directory = ROOT / '.local/wheelhouse'
    directory.mkdir(parents=True, exist_ok=True)
    doc = sources()
    for package, item in doc.items():
        path = directory / item['filename']
        expected = item['expected_sha256']
        if path.exists() and expected:
            with path.open('rb') as stream:
                actual = checksum(stream)
            if actual == expected:
                item['sha256'] = actual
                continue
        elif path.exists():
            request = urllib.request.Request(item['url'], method='HEAD')
            with urllib.request.urlopen(request, timeout=120) as response:
                remote_size = int(response.headers['Content-Length'])
            if path.stat().st_size == remote_size:
                with zipfile.ZipFile(path) as wheel:
                    if wheel.testzip() is not None:
                        raise ValueError(f'Corrupt official wheel: {package}')
                with path.open('rb') as stream:
                    item['sha256'] = checksum(stream)
                continue
        subprocess.run(['curl.exe', '--fail', '--location', '--retry', '8', '--retry-all-errors',
            '--connect-timeout', '30', '--continue-at', '-', '--output', str(path), item['url']], check=True)
        with path.open('rb') as stream:
            actual = checksum(stream)
        if expected and actual != expected:
            raise ValueError(f'Incomplete/invalid official wheel: {package}; SHA256 verification failed')
        item['sha256'] = actual
    (directory / 'runtime-wheels.json').write_text(json.dumps(doc, indent=2), encoding='utf-8')
    verify(directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', action='store_true')
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.sources:
        print(json.dumps(sources(), indent=2))
    elif args.verify:
        verify(args.verify)
    else:
        fetch()
