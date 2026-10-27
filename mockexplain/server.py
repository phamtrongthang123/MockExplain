"""Small loopback-only HTTP API. Inputs live only for the inspection request."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import socket
from urllib.parse import unquote, urlsplit

from .inspect import inspect_files
from .limits import InspectionError, MAX_UPLOAD

STATIC = Path(__file__).with_name('static')
ASSETS = {'/': ('index.html', 'text/html; charset=utf-8'),
          '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
          '/state.mjs': ('state.mjs', 'text/javascript; charset=utf-8'),
          '/style.css': ('style.css', 'text/css; charset=utf-8')}


class Handler(BaseHTTPRequestHandler):
    server_version = 'MockExplain/0.1'

    def setup(self):
        super().setup()
        self.connection.settimeout(20)

    def local_request(self):
        port = self.server.server_address[1]
        allowed = {f'http://127.0.0.1:{port}', f'http://localhost:{port}'}
        host = self.headers.get('Host', '')
        origin = self.headers.get('Origin')
        return f'http://{host}' in allowed and (origin is None or origin in allowed)

    def send(self, code, data, mime='application/json; charset=utf-8'):
        if not isinstance(data, bytes):
            data = json.dumps(data, allow_nan=False, separators=(',', ':')).encode()
        self.send_response(code)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass  # A browser may switch away while an upload is finishing.

    def do_GET(self):
        if not self.local_request():
            return self.send(403, {'error': 'Use the local MockExplain address printed by the server.'})
        path = urlsplit(self.path).path
        if path == '/api/health':
            return self.send(200, {'app': 'MockExplain', 'version': '0.1.0'})
        if path not in ASSETS:
            return self.send(404, {'error': 'Not found.'})
        filename, mime = ASSETS[path]
        self.send(200, (STATIC / filename).read_bytes(), mime)

    def do_POST(self):
        if not self.local_request():
            return self.send(403, {'error': 'Only same-origin local requests are accepted.'})
        if urlsplit(self.path).path != '/api/inspect':
            return self.send(404, {'error': 'Not found.'})
        try:
            length = int(self.headers.get('Content-Length', '-1'))
            if not 0 < length <= MAX_UPLOAD:
                return self.send(413, {'error': 'Upload must be nonempty and at most 128 MiB combined.'})
            if self.headers.get('Transfer-Encoding') or self.headers.get('Content-Type') != 'application/octet-stream':
                raise InspectionError('Use the app upload control to send model files.')
            header = self.headers.get('X-MockExplain-Files', '')
            if len(header) > 8192:
                raise InspectionError('Too much upload file metadata.')
            manifest = json.loads(unquote(header))
            if not isinstance(manifest, list) or not 1 <= len(manifest) <= 3:
                raise InspectionError('Select up to three files: one .pt, one .py, and one optional JSON config.')
            if any(not isinstance(f, dict) or not isinstance(f.get('name'), str) or type(f.get('size')) is not int or f['size'] <= 0 for f in manifest):
                raise InspectionError('Invalid upload file metadata.')
            if sum(f['size'] for f in manifest) != length:
                raise InspectionError('Upload length does not match file metadata.')
            files = []
            for f in manifest:
                data = self.rfile.read(f['size'])
                if len(data) != f['size']:
                    raise InspectionError('Upload was interrupted. Select the file again.')
                files.append((f['name'], data))
            report = inspect_files(files)
            self.send(200, report)
        except (InspectionError, ValueError, UnicodeError, RecursionError) as exc:
            message = str(exc) if isinstance(exc, InspectionError) else 'Invalid upload metadata. Select the file again using the upload control.'
            self.send(422, {'error': message})
        except (TimeoutError, socket.timeout):
            self.send(408, {'error': 'Upload timed out. Try a smaller local file.'})

    def log_message(self, fmt, *args):
        # Avoid displaying untrusted filenames or uploaded content in the terminal.
        if args and str(args[1] if len(args) > 1 else '') not in ('200',):
            super().log_message(fmt, *args)


def make_server(port=8000):
    return HTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description='MockExplain: local structural insights and interpretability recommendations.')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if not 0 < args.port < 65536:
        parser.error('Port must be between 1 and 65535.')
    try:
        server = make_server(args.port)
    except OSError as exc:
        parser.exit(1, f'Cannot start MockExplain: {exc}. Try --port 8001.\n')
    print(f'MockExplain v0.1 — http://127.0.0.1:{server.server_address[1]}', flush=True)
    print('Local inspection only. Press Ctrl+C to stop.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
