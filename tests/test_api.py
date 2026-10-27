import http.client
import json
from pathlib import Path
import threading
import unittest
from urllib.parse import quote

from mockexplain.server import make_server

ROOT = Path(__file__).resolve().parents[1]


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=5)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        result = response.status, response.read(), dict(response.getheaders())
        connection.close()
        return result

    def upload(self, name, data):
        return self.request('POST', '/api/inspect', data, {'Content-Type': 'application/octet-stream', 'X-MockExplain-Files': quote(json.dumps([{'name': name, 'size': len(data)}]))})

    def test_real_http_uploads_and_independent_identity(self):
        results = []
        for name in ['cnn.py', 'transformer.py', 'tiny_cnn.pt', 'unknown.py']:
            code, body, _ = self.upload(name, (ROOT / 'examples' / name).read_bytes())
            self.assertEqual(code, 200, body)
            results.append(json.loads(body))
        self.assertEqual(len({p['id'] for p in results}), 4)
        self.assertIn('CNN', results[0]['inferences']['family'])
        self.assertIn('Language', results[1]['inferences']['family'])
        self.assertEqual(results[3]['recommendations'], [])

    def test_errors_and_same_origin_limits(self):
        code, body, _ = self.upload('bad.py', b'def (')
        self.assertEqual(code, 422); self.assertIn('parse Python', json.loads(body)['error'])
        code, body, _ = self.upload('bad.pt', b'not weights')
        self.assertEqual(code, 422); self.assertIn('state_dict', json.loads(body)['error'])
        self.assertEqual(self.request('POST', '/api/inspect', b'x', {'Origin': 'https://untrusted.example'})[0], 403)
        self.assertEqual(self.request('GET', '/', headers={'Host': 'untrusted.example'})[0], 403)
        self.assertEqual(self.request('GET', '/../pyproject.toml')[0], 404)
        self.assertEqual(self.request('POST', '/api/inspect', b'')[0], 413)

    def test_installed_assets_security_headers_and_no_prompt_form(self):
        for path in ['/', '/app.js', '/state.mjs', '/style.css']:
            code, body, headers = self.request('GET', path)
            self.assertEqual(code, 200)
            self.assertEqual(headers['X-Content-Type-Options'], 'nosniff')
            self.assertIn("object-src 'none'", headers['Content-Security-Policy'])
            self.assertTrue(body)
        html = self.request('GET', '/')[1].decode()
        self.assertIn('Model overview', html); self.assertIn('Suggested methods', html)
        self.assertNotIn('textarea', html)


if __name__ == '__main__': unittest.main()
