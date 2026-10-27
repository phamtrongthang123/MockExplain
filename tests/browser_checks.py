"""Real Chromium workflow check. Optional dependency: pip install playwright.

Run: python tests/browser_checks.py
Use CHROMIUM_EXECUTABLE to select an already installed Chromium, if needed.
"""
import asyncio
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import threading
import zipfile

from playwright.async_api import async_playwright, expect

from mockexplain.server import make_server

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('MOCKEXPLAIN_BROWSER_ARTIFACTS', '/tmp/mockexplain-browser'))


async def check():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    server = make_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors = []
    posts = []
    try:
        async with async_playwright() as pw:
            options = {'headless': True}
            if os.environ.get('CHROMIUM_EXECUTABLE'):
                options['executable_path'] = os.environ['CHROMIUM_EXECUTABLE']
            browser = await pw.chromium.launch(**options)
            page = await browser.new_page(viewport={'width': 1440, 'height': 1100}, accept_downloads=True)
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda request: posts.append(request.url) if request.method == 'POST' else None)
            await page.goto(f'http://127.0.0.1:{server.server_address[1]}')
            await expect(page.get_by_text('Upload a .pt or .py file.')).to_be_visible()
            assert await page.locator('textarea, input[type=text]').count() == 0
            await page.screenshot(path=str(ARTIFACTS / 'empty-desktop.png'), full_page=True)

            async def upload(path):
                await page.locator('#files').set_input_files(str(ROOT / path))
                await expect(page.locator('#models option:checked')).to_have_text(Path(path).name)
                await expect(page.locator('#cards')).to_have_attribute('aria-busy', 'false')

            await upload('examples/cnn.py')
            await expect(page.locator('#overview .family')).to_contain_text('CNN')
            await expect(page.locator('#methods')).to_contain_text('Grad-CAM')
            paper = page.locator('#methods a[href="https://arxiv.org/abs/1610.02391"]')
            await expect(paper).to_be_visible()
            await expect(page.locator('#methods details').nth(1)).not_to_have_attribute('open', '')
            await page.get_by_text('Details', exact=True).nth(1).click()
            await expect(page.locator('#methods')).to_contain_text('feature maps and gradients')

            # Delay a real HTTP upload to verify loading + switching + late completion in DOM.
            release = asyncio.Event()
            async def delayed(route):
                await release.wait()
                await route.continue_()
            await page.route('**/api/inspect', delayed)
            await page.locator('#files').set_input_files(str(ROOT / 'examples/transformer.py'))
            await expect(page.locator('#status')).to_contain_text('Inspecting transformer.py')
            await expect(page.locator('#methods')).not_to_contain_text('Grad-CAM')
            await expect(page.locator('#export-json')).to_be_disabled()
            await page.locator('#models').select_option('1')
            await expect(page.locator('#overview .family')).to_contain_text('CNN')
            release.set()
            await expect(page.locator('#models option[value="2"]')).not_to_contain_text('inspecting')
            await expect(page.locator('#overview .family')).to_contain_text('CNN')
            await page.unroute('**/api/inspect', delayed)
            await page.locator('#models').select_option('2')
            await expect(page.locator('#overview .family')).to_contain_text('Language transformer')
            await expect(page.locator('#methods')).to_contain_text('J-lens')
            await expect(page.locator('#methods a[href="https://arxiv.org/abs/2607.15495"]')).to_be_visible()
            await expect(page.locator('#methods')).not_to_contain_text('Grad-CAM')
            async with page.expect_download() as download:
                await page.locator('#export-json').click()
            transformer_report = ARTIFACTS / 'transformer-report.json'
            await (await download.value).save_as(str(transformer_report))
            await page.screenshot(path=str(ARTIFACTS / 'transformer-desktop.png'), full_page=True)

            await upload('examples/tiny_cnn.pt')
            await page.get_by_text('Tensors (9)', exact=True).click()
            await expect(page.locator('#overview')).to_contain_text('[8, 3, 3, 3]')
            await expect(page.locator('#overview')).to_contain_text('norm.num_batches_tracked')
            await page.get_by_text('Weight samples (9 tensors)', exact=True).click()
            await expect(page.locator('#overview')).to_contain_text('Sample min / max')
            await page.get_by_text('Inspection details', exact=True).click()
            await expect(page.locator('#overview')).to_contain_text('293 of 293 tensor values sampled')
            async with page.expect_download() as download:
                await page.locator('#export-json').click()
            cnn_report = ARTIFACTS / 'cnn-report.json'
            await (await download.value).save_as(str(cnn_report))
            saved = json.loads(cnn_report.read_text())
            assert saved['inputs'][0]['sha256'] == hashlib.sha256((ROOT / 'examples/tiny_cnn.pt').read_bytes()).hexdigest()
            assert saved['coverage']['tensors_inspected'] == 9
            assert saved['rule_version'] == 'mockexplain-rules-0.1.1'
            async with page.expect_download() as download:
                await page.locator('#export-md').click()
            md_path = ARTIFACTS / 'cnn-report.md'
            await (await download.value).save_as(str(md_path))
            for text in ['Input identity', 'SHA-256', 'Inspection coverage', 'Prerequisites', 'Limitations', 'https://arxiv.org/abs/1610.02391']:
                assert text in md_path.read_text(), text
            await page.screenshot(path=str(ARTIFACTS / 'cnn-desktop.png'), full_page=True)

            before = len(posts)
            await page.locator('#compare').click()
            await page.locator('#compare-right').select_option('2')
            await expect(page.locator('#comparison-content')).to_contain_text('Language transformer candidate')
            await expect(page.locator('#comparison-content')).to_contain_text('Grad-CAM')
            await expect(page.locator('#comparison-content')).to_contain_text('J-lens')
            assert len(posts) == before
            await page.screenshot(path=str(ARTIFACTS / 'comparison-desktop.png'), full_page=True)
            await page.locator('#close-comparison').click()

            # Attachments produce a new identity without mutating the earlier profile.
            await page.locator('#attachment').set_input_files([str(ROOT / 'examples/cnn.py'), str(ROOT / 'examples/config.json')])
            await expect(page.locator('#cards')).to_have_attribute('aria-busy', 'false')
            assert await page.locator('#models').input_value() == '4'
            await page.get_by_text('Configuration', exact=True).click()
            await expect(page.locator('#overview')).to_contain_text('hidden_size')
            await page.locator('#models').select_option('3')
            await expect(page.locator('#overview')).not_to_contain_text('Configuration')

            await upload('examples/unknown.py')
            await expect(page.locator('#methods')).to_contain_text('No suggestions for this model yet')
            await page.locator('#files').set_input_files({'name': 'broken.py', 'mimeType': 'text/plain', 'buffer': b'def ('})
            await expect(page.locator('#status')).to_contain_text('Cannot parse Python')
            await expect(page.locator('#methods')).not_to_contain_text('Grad-CAM')
            await expect(page.locator('#export-json')).to_be_disabled()

            with tempfile.TemporaryDirectory() as tmp:
                oversized = Path(tmp) / 'oversized.pt'
                with oversized.open('wb') as stream:
                    stream.truncate(128 * 1048576 + 1)
                before = len(posts)
                await page.locator('#files').set_input_files(str(oversized))
                await expect(page.locator('#status')).to_contain_text('Combined upload exceeds 128 MiB')
                assert len(posts) == before

            with tempfile.TemporaryDirectory() as tmp:
                sentinel = Path(tmp) / 'executed'
                class Evil:
                    def __reduce__(self):
                        return os.system, (f'touch {sentinel}',)
                stream = io.BytesIO()
                with zipfile.ZipFile(stream, 'w') as z:
                    z.writestr('model/data.pkl', pickle.dumps({'weight': Evil()}, protocol=2))
                await page.locator('#files').set_input_files({'name': 'unsafe.pt', 'mimeType': 'application/octet-stream', 'buffer': stream.getvalue()})
                await expect(page.locator('#status')).to_contain_text('Unsupported or unsafe checkpoint object')
                assert not sentinel.exists()
                await page.screenshot(path=str(ARTIFACTS / 'error-desktop.png'), full_page=True)

            # A model folder may include unrelated notes; only supported files are selected.
            with tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                (folder / 'cnn.py').write_bytes((ROOT / 'examples/cnn.py').read_bytes())
                (folder / 'tiny_cnn.pt').write_bytes((ROOT / 'examples/tiny_cnn.pt').read_bytes())
                (folder / 'README.txt').write_text('Unrelated notes are not uploaded.')
                await page.locator('#folder').set_input_files(str(folder))
                await expect(page.locator('#cards')).to_have_attribute('aria-busy', 'false')
                await expect(page.locator('#overview .family')).to_contain_text('CNN')
                await page.get_by_text('Files', exact=True).click()
                await expect(page.locator('#overview')).not_to_contain_text('README.txt')

            # Saved reports survive reload by explicit import, without reinspection.
            await page.reload()
            await expect(page.locator('#overview')).to_contain_text('Upload a .pt or .py file.')
            before = len(posts)
            await page.locator('#report-file').set_input_files([str(cnn_report), str(transformer_report)])
            await expect(page.locator('#status')).to_contain_text('Saved report')
            assert await page.locator('#models option').count() == 2
            assert len(posts) == before
            await expect(page.locator('#overview .family')).to_contain_text('Language transformer')
            await page.locator('#compare').click()
            await expect(page.locator('#comparison-content')).to_contain_text('Grad-CAM')
            await page.locator('#close-comparison').click()
            await page.locator('#models').select_option('1')
            await page.set_viewport_size({'width': 390, 'height': 844})
            await expect(page.locator('#overview .family')).to_contain_text('CNN')
            assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            await page.screenshot(path=str(ARTIFACTS / 'cnn-mobile.png'), full_page=True)
            # Invalid saved reports must show their own error state.
            await page.locator('#report-file').set_input_files({'name': 'invalid.json', 'mimeType': 'application/json', 'buffer': b'{"schema_version":1}'})
            await expect(page.locator('#status')).to_contain_text('Unsupported or malformed report')
            await expect(page.locator('#methods')).not_to_contain_text('Grad-CAM')
            assert not errors, errors
            await browser.close()
    finally:
        server.shutdown(); server.server_close(); thread.join()
    print('PASS: real browser uploads (CNN/transformer source + tensor checkpoint), loading/switch races, attachments/folder selection, unknown/malformed/unsafe inputs, JSON/Markdown downloads, report reimport/comparison, desktop/mobile. No browser errors.')
    print(f'Browser artifacts: {ARTIFACTS}')


if __name__ == '__main__':
    asyncio.run(check())
