import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from mockexplain.checkpoint import MetadataParser, Tensor, Storage, inspect_checkpoint
from mockexplain.inspect import inspect_files
from mockexplain.limits import InspectionError, MAX_SAMPLE_BYTES
from mockexplain.rules import RULE_VERSION

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures'


def report(path):
    path = ROOT / path
    return inspect_files([(path.name, path.read_bytes())])


def archive(metadata, storage=None, compression=zipfile.ZIP_STORED):
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', compression=compression) as z:
        z.writestr('model/data.pkl', metadata)
        z.writestr('model/byteorder', 'little')
        if storage is not None:
            z.writestr('model/data/0', storage)
    return out.getvalue()


def rewrite(data, transform):
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(out, 'w') as dst:
        for entry in src.infolist():
            value = src.read(entry)
            value = transform(entry.filename, value)
            if value is not None:
                dst.writestr(entry.filename, value)
    return out.getvalue()


class ReaderTests(unittest.TestCase):
    def test_real_state_dict_metadata_buffers_and_no_parameter_claim(self):
        p = report('examples/tiny_cnn.pt')
        rows = {t['name']: t for t in p['facts']['tensors']}
        self.assertEqual(rows['conv.weight']['shape'], [8, 3, 3, 3])
        self.assertEqual(rows['conv.weight']['dtype'], 'float32')
        self.assertEqual(rows['norm.num_batches_tracked']['dtype'], 'int64')
        self.assertEqual(p['facts']['tensor_element_count'], 293)
        self.assertIsNone(p['facts']['parameter_count'])
        self.assertEqual(p['coverage']['tensors_inspected'], 9)
        self.assertEqual(p['coverage']['sampled_values'], 293)

    def test_real_protocols_dtypes_noncontiguous_offsets_aliases_empty_nonfinite(self):
        for protocol in (2, 4, 5):
            with self.subTest(protocol=protocol):
                p = report(f'tests/fixtures/tensors_p{protocol}.pt')
                rows = {t['name']: t for t in p['facts']['tensors']}
                self.assertEqual(rows['view']['stride'], [1, 6])
                self.assertEqual(rows['view']['statistics']['mean'], 11.5)
                self.assertEqual(rows['offset']['statistics']['min'], 8)
                self.assertEqual(rows['offset']['statistics']['max'], 23)
                self.assertEqual(rows['bfloat']['statistics']['mean'], .75)
                self.assertEqual(rows['half']['statistics']['min'], -1.)
                self.assertEqual(rows['bool']['statistics']['zero_fraction'], 1 / 3)
                self.assertIsNone(rows['empty']['statistics'])
                self.assertEqual(rows['nonfinite']['statistics']['nonfinite_values'], 2)
                self.assertEqual(rows['view']['storage_key'], rows['alias']['storage_key'])
                self.assertEqual(p['facts']['storage_bytes'], 8 + 3 + 6 + 40 + 12 + 96)
                json.dumps(p, allow_nan=False)
        self.assertEqual(report('tests/fixtures/wrapped.pt')['facts']['checkpoint_wrapper'], 'state_dict')

    def test_metadata_first_bounded_raw_reads_and_labeled_partial_coverage(self):
        import mockexplain.checkpoint as reader
        raw = (FIXTURES / 'partial.pt').read_bytes()
        reads = []
        original = reader.storage_view

        class GuardedView:
            def __init__(self, view): self.view = view
            def __getitem__(self, index):
                self.assert_slice(index)
                reads.append(index.stop - index.start)
                return self.view[index]
            @staticmethod
            def assert_slice(index):
                if not isinstance(index, slice) or index.stop - index.start > 8:
                    raise AssertionError('Unbounded weight read')

        with patch.object(reader, 'storage_view', side_effect=lambda d, e: GuardedView(original(d, e))):
            p = inspect_files([('partial.pt', raw)])
        self.assertEqual(p['coverage']['tensors_inspected'], 40)
        self.assertEqual(p['coverage']['tensors_sampled'], 32)
        self.assertEqual(p['coverage']['sampled_values'], 8192)
        self.assertEqual(sum(reads), p['coverage']['sample_bytes'])
        self.assertLessEqual(sum(reads), MAX_SAMPLE_BYTES)
        self.assertTrue(p['coverage']['partial'])
        self.assertEqual(p['coverage']['total_tensor_values'], 40960)
        self.assertEqual(sum(t['statistics'] is None for t in p['facts']['tensors']), 8)
        self.assertEqual(p['facts']['tensors'][0]['statistics']['max'], 1023)
        # A missing storage late in the archive must reject before any sampling.
        broken = rewrite(raw, lambda n, v: None if n.endswith('/data/39') else v)
        with patch.object(reader, 'sample_tensor', side_effect=AssertionError('sampled before metadata validation')):
            with self.assertRaisesRegex(InspectionError, 'Missing raw tensor storage'):
                inspect_checkpoint(broken, 'broken.pt')

    def test_pickle_reducer_does_not_execute(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'executed'
            class Evil:
                def __reduce__(self):
                    return os.system, (f'touch {target}',)
            payload = pickle.dumps({'weight': Evil()}, protocol=2)
            with self.assertRaisesRegex(InspectionError, 'Unsupported or unsafe checkpoint object'):
                inspect_checkpoint(archive(payload), 'unsafe.pt')
            self.assertFalse(target.exists())
            with self.assertRaisesRegex(InspectionError, 'Legacy raw pickles'):
                inspect_checkpoint(payload, 'legacy.pt')
            self.assertFalse(target.exists())

    def test_malformed_unsupported_and_oversized_metadata(self):
        cases = [b'not a zip', archive(b'bad pickle'), archive(pickle.dumps({'weight': [1, 2, 3]})),
                 archive(b'x' * (2 * 1024 * 1024 + 1)), archive(pickle.dumps({}), compression=zipfile.ZIP_DEFLATED)]
        for data in cases:
            with self.subTest(size=len(data)), self.assertRaises(InspectionError):
                inspect_checkpoint(data, 'bad.pt')
        data = (ROOT / 'examples/tiny_cnn.pt').read_bytes()
        # Wrong byte order and short storage get useful errors.
        for transform in (lambda n, v: b'invalid' if n.endswith('/byteorder') else v,
                          lambda n, v: v[:-1] if n.endswith('/data/0') else v):
            with self.assertRaises(InspectionError):
                inspect_checkpoint(rewrite(data, transform), 'bad.pt')
        duplicate = io.BytesIO()
        with zipfile.ZipFile(duplicate, 'w') as z:
            z.writestr('../data.pkl', b'N.')
        with self.assertRaisesRegex(InspectionError, 'unsafe ZIP paths'):
            inspect_checkpoint(duplicate.getvalue(), 'bad.pt')

    def test_cuda_location_is_metadata_only(self):
        data = (ROOT / 'examples/tiny_cnn.pt').read_bytes()
        data = rewrite(data, lambda n, v: v.replace(b'X\x03\x00\x00\x00cpu', b'X\x06\x00\x00\x00cuda:0') if n.endswith('/data.pkl') else v)
        result = inspect_checkpoint(data, 'cuda.pt')
        self.assertTrue(all(t['device_recorded'] == 'cuda:0' for t in result['tensors']))
        self.assertEqual(result['sampled_values'], 293)

    def test_parser_rejects_resource_limits_and_bad_shapes(self):
        with patch('mockexplain.checkpoint.MAX_OPS', 1):
            with self.assertRaisesRegex(InspectionError, 'parser limits'):
                MetadataParser().parse(pickle.dumps({'x': 1}, protocol=2))
        parser = MetadataParser()
        from mockexplain.checkpoint import Symbol
        for shape, stride in [((-1,), (1,)), ((2**49,), (1,)), ((2,), (-1,)), ((2,), ())]:
            with self.assertRaises(InspectionError):
                parser.reduce(Symbol('torch._utils', '_rebuild_tensor'), (Storage('0', 'FloatStorage', 4, 'cuda:0'), 0, shape, stride))

    def test_no_heavy_imports_or_pickle_fallback(self):
        import subprocess
        result = subprocess.run([os.sys.executable, '-c', "import sys; from mockexplain.inspect import inspect_files; assert {'torch','numpy','transformers'}.isdisjoint(sys.modules)"], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for path in (ROOT / 'mockexplain').glob('*.py'):
            text = path.read_text()
            self.assertNotIn('pickle.load(', text)
            self.assertNotIn('torch.load(', text)


class SourceAndRuleTests(unittest.TestCase):
    def test_source_never_executes_imports_calls_or_top_level_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp) / 'source_executed'
            source = f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\nimport nonexistent_upload_module\nfrom torch.nn import Conv2d as C\nlayer = C(3, 8, 3)\n".encode()
            p = inspect_files([('evil.py', source)])
            self.assertFalse(sentinel.exists())
            self.assertEqual(p['facts']['declarations'][0]['constructor'], 'torch.nn.Conv2d')
            self.assertIsNone(p['facts']['tensor_element_count'])

    def test_unknown_malformed_and_bounded_source(self):
        self.assertEqual(report('examples/unknown.py')['recommendations'], [])
        self.assertEqual(inspect_files([('unknown.py', b'# no modules\nx = 3')])['inferences']['family'], 'Unknown / not covered')
        for name, data in [('broken.py', b'def ('), ('broken.py', b'\xff'), ('large.py', b'#' * (512 * 1024 + 1)), ('bad.pt', b''), ('bad.exe', b'123')]:
            with self.subTest(name=name), self.assertRaises(InspectionError): inspect_files([(name, data)])
        p = inspect_files([('literal.py', b'from torch import nn\nx = nn.Linear(1e999, 2)')])
        json.dumps(p, allow_nan=False)

    def test_deterministic_justified_rules_and_hybrid(self):
        cnn = report('examples/cnn.py')
        language = report('examples/transformer.py')
        self.assertEqual([r['id'] for r in cnn['recommendations']], ['filter-inspection', 'grad-cam'])
        self.assertEqual([r['id'] for r in language['recommendations']], ['jacobian-lens', 'logit-lens', 'sae'])
        self.assertEqual(language, report('examples/transformer.py'))
        self.assertEqual(language['rule_version'], RULE_VERSION)
        self.assertIn('no separately trained lens', ' '.join(language['recommendations'][1]['prerequisites']))
        self.assertIn('pre-fitted', ' '.join(language['recommendations'][0]['prerequisites']))
        source = (ROOT / 'examples/transformer.py').read_bytes() + b'\nimage_conv = nn.Conv2d(3, 8, 3)\n'
        hybrid = inspect_files([('hybrid.py', source)])
        self.assertIn('Hybrid', hybrid['inferences']['family'])
        self.assertEqual(len(hybrid['recommendations']), 4)
        for r in [*cnn['recommendations'], *language['recommendations']]:
            self.assertTrue(r['reason'] and r['prerequisites'] and r['limitations'] and r['references'])
            self.assertLessEqual(len(r['purpose'].split()), 12)
            self.assertTrue(all(ref['url'].startswith('https://') for ref in r['references']))
        self.assertEqual(language['recommendations'][0]['references'][0]['url'], 'https://arxiv.org/abs/2607.15495')
        encoder = inspect_files([('encoder.py', b'from torch import nn\nx = nn.TransformerEncoderLayer(16, 2)')])
        self.assertEqual([r['id'] for r in encoder['recommendations']], ['sae', 'activation-analysis'])

    def test_input_identity_attachment_and_config(self):
        pt = ('tiny_cnn.pt', (ROOT / 'examples/tiny_cnn.pt').read_bytes())
        py = ('cnn.py', (ROOT / 'examples/cnn.py').read_bytes())
        a = inspect_files([pt]); b = inspect_files([pt, py]); c = inspect_files([py, pt])
        self.assertNotEqual(a['id'], b['id'])
        self.assertEqual(b, c)
        self.assertEqual(a['inputs'][0]['sha256'], hashlib.sha256(pt[1]).hexdigest())
        self.assertTrue(any('not checked' in s for s in b['coverage']['notes']))
        config = ('config.json', (ROOT / 'examples/config.json').read_bytes())
        self.assertEqual(inspect_files([py, config])['facts']['config']['fields']['hidden_size'], 16)
        for files in [[config], [py, py], [('bad.json', b'{oops'), py], [('nan.json', b'{"vocab_size":NaN}'), py], [('inf.json', b'{"vocab_size":1e999}'), py]]:
            with self.assertRaises(InspectionError): inspect_files(files)


if __name__ == '__main__':
    unittest.main()
