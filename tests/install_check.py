"""Verify an installed package from outside the checkout (no runtime extras).

Usage: python tests/install_check.py /path/to/fresh-env/bin/python
"""
import http.client
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]


def check(python):
    with tempfile.TemporaryDirectory() as workdir:
        probe = subprocess.run([python, '-I', '-c', "from importlib.metadata import distribution; from importlib.resources import files; import sys, mockexplain; assert not distribution('mockexplain').requires; assert {'torch','numpy','transformers'}.isdisjoint(sys.modules); assert all(files('mockexplain').joinpath('static', p).is_file() for p in ('index.html','app.js','state.mjs','style.css')); print(mockexplain.__file__)"], cwd=workdir, capture_output=True, text=True, check=True)
        assert str(ROOT / 'mockexplain') not in probe.stdout, 'This must be a regular installation, not editable.'
        print('Installed package:', probe.stdout.strip())
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        process = subprocess.Popen([python, '-I', '-m', 'mockexplain', '--port', str(port)], cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            for _ in range(100):
                try:
                    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
                    conn.request('GET', '/api/health')
                    response = conn.getresponse()
                    assert response.status == 200
                    response.read(); conn.close()
                    break
                except OSError:
                    if process.poll() is not None:
                        raise AssertionError(process.communicate())
                    time.sleep(.05)
            else:
                raise AssertionError('Installed server never became ready')
            for path in ['/', '/app.js', '/state.mjs', '/style.css']:
                conn = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
                conn.request('GET', path); response = conn.getresponse()
                assert response.status == 200 and response.read(), path
                conn.close()
            for name in ['cnn.py', 'transformer.py', 'tiny_cnn.pt']:
                data = (ROOT / 'examples' / name).read_bytes()
                conn = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
                conn.request('POST', '/api/inspect', data, {'Content-Type': 'application/octet-stream', 'X-MockExplain-Files': quote(json.dumps([{'name': name, 'size': len(data)}]))})
                response = conn.getresponse(); result = json.loads(response.read()); conn.close()
                assert response.status == 200 and result['recommendations'], result
        finally:
            process.terminate()
            process.communicate(timeout=5)
    print('PASS: fresh dependency-free installed package launched outside checkout; all assets and three real uploads work.')


if __name__ == '__main__':
    check(str(Path(sys.argv[1]).absolute()) if len(sys.argv) > 1 else sys.executable)
