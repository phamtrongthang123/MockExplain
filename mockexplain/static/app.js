import {ProfileStore} from './state.mjs';

const $ = id => document.getElementById(id);
const node = (tag, text, className) => {
  const el = document.createElement(tag);
  if (text !== undefined && text !== null) el.textContent = String(text);
  if (className) el.className = className;
  return el;
};
const number = value => value == null ? 'Unknown' : Number(value).toLocaleString();
const bytes = n => n == null ? 'Unknown' : n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KiB` : `${(n / 1048576).toFixed(1)} MiB`;
const store = new ProfileStore(render);

function list(items) { const el = node('ul'); for (const item of items) el.append(node('li', item)); return el; }
function details(title, ...children) { const el = node('details'); el.append(node('summary', title), ...children); return el; }
function table(headers, rows) {
  const wrap = node('div', null, 'table-wrap'); const el = node('table'); const head = node('thead');
  const tr = node('tr'); for (const h of headers) tr.append(node('th', h)); head.append(tr); el.append(head);
  const body = node('tbody');
  for (const row of rows) { const tr = node('tr'); for (const text of row) tr.append(node('td', text)); body.append(tr); }
  el.append(body); wrap.append(el); return wrap;
}
function metric(title, value) { const el = node('div', null, 'metric'); el.append(node('strong', value), node('span', title)); return el; }
function message(text, state = '') { $('status').className = state === 'ready' ? 'visually-hidden' : `status ${state}`; $('status').replaceChildren(); if (state === 'loading') $('status').append(node('span', '', 'spinner')); $('status').append(document.createTextNode(text)); }
function label(profile) { return profile.inputs.map(i => i.name).join(' + '); }

function render() {
  const active = store.active;
  $('models').replaceChildren();
  if (!store.profiles.length) $('models').append(new Option('No model uploaded', ''));
  for (const entry of store.profiles) $('models').append(new Option(`${entry.label}${entry.status === 'loading' ? ' · inspecting' : entry.status === 'error' ? ' · error' : ''}`, entry.key));
  $('models').value = active?.key || '';
  $('toolbar').hidden = !active;
  const ready = active?.status === 'ready';
  for (const id of ['export-json', 'export-md']) $(id).disabled = !ready;
  $('attach').disabled = !ready || !active.files;
  $('compare').disabled = store.profiles.filter(p => p.status === 'ready').length < 2;
  $('overview').replaceChildren(); $('methods').replaceChildren();
  $('cards').setAttribute('aria-busy', active?.status === 'loading' ? 'true' : 'false');
  if (!active) {
    message('', 'ready');
    $('overview').append(node('p', 'Upload a .pt or .py file.', 'hint'));
    $('methods').append(node('p', 'Suggestions will appear here.', 'hint'));
    return;
  }
  if (active.status === 'loading') {
    message(`Inspecting ${active.label}…`, 'loading');
    return;
  }
  if (active.status === 'error') {
    message(`${active.label}: ${active.error}`, 'error');
    $('overview').append(node('p', 'Could not inspect this file.', 'hint'));
    $('methods').append(node('p', 'No suggestions.', 'hint'));
    return;
  }
  const p = active.profile;
  message(active.imported ? 'Saved report; original files not checked.' : 'Inspection complete.', active.imported ? '' : 'ready');
  renderOverview(p); renderMethods(p);
}

function renderOverview(p) {
  const out = $('overview'); const f = p.facts; const c = p.coverage;
  out.append(node('div', p.inferences.family, 'family'));
  const metrics = node('div', null, 'metrics');
  if (f.tensors.length) metrics.append(metric('Tensor elements', number(f.tensor_element_count)), metric('Trainable parameters', number(f.parameter_count)), metric('Storage', bytes(f.storage_bytes)), metric('Dtypes', f.dtypes.join(', ') || 'Unknown'));
  else metrics.append(metric('Source call sites', number(f.declarations.length)), metric('Weights', 'Not provided'));
  out.append(metrics);
  out.append(details('Files', table(['File', 'Size', 'SHA-256'], p.inputs.map(i => [i.name, bytes(i.bytes), i.sha256])), node('p', `Profile ID: ${p.id}`, 'hint')));
  if (f.tensors.length) {
    out.append(details(`Tensors (${f.tensors.length})`, table(['Tensor key', 'Shape / stride', 'Dtype', 'Elements', 'Recorded device'], f.tensors.map(t => [t.name, `[${t.shape.join(', ')}] / [${t.stride.join(', ')}]`, t.dtype, number(t.numel), t.device_recorded]))));
    const sampled = f.tensors.filter(t => t.statistics);
    out.append(details(`Weight samples (${sampled.length} tensors)`, node('p', 'Up to 256 evenly spaced values from each of 32 tensors. Samples do not describe all weights.'), table(['Tensor', 'Values sampled / total', 'Sample min / max', 'Sample mean', 'Sample zeros / nonfinite'], sampled.map(t => {
      const s = t.statistics; const fmt = v => v == null ? 'No finite values' : Number(v).toPrecision(5);
      return [t.name, `${number(s.sampled_values)} / ${number(s.total_values)}`, `${fmt(s.min)} / ${fmt(s.max)}`, fmt(s.mean), `${(s.zero_fraction * 100).toFixed(1)}% / ${s.nonfinite_values}`];
    }))));
  }
  if (f.declarations.length || f.classes.length) out.append(details(`Source declarations (${f.declarations.length} call sites)`, node('p', `Classes: ${f.classes.join(', ') || 'None'}. Call sites do not establish runtime layers or order.`), table(['Declaration', 'Constructor', 'Literal arguments', 'Source line'], f.declarations.map(d => [d.name, d.constructor, JSON.stringify({args: d.args, keywords: d.keywords}), `${d.source}:${d.line}`]))));
  if (f.config) out.append(details('Configuration', node('pre', JSON.stringify(f.config.fields, null, 2)), node('p', f.config.note)));
  out.append(details('Structural clues', node('p', p.inferences.note), p.inferences.evidence.length ? list(p.inferences.evidence.map(e => `${e.source}: ${e.evidence}`)) : node('p', 'No recognized clues.')));
  out.append(details('Inspection details', node('p', c.metadata), node('p', f.parameter_count_note), list([
    `${number(c.tensors_inspected)} tensor metadata entries; ${number(c.tensors_sampled)} tensors sampled.`,
    `${number(c.sampled_values)} of ${number(c.total_tensor_values)} tensor values sampled; ${bytes(c.sample_bytes)} of raw sample bytes.`,
    `Limits: ${c.sample_limit.tensors} tensors × ${c.sample_limit.values_per_tensor} values; ${bytes(c.sample_limit.bytes)} maximum sample bytes.`, c.partial_reason, ...c.notes]), node('p', `Reader: ${p.reader_version} · Rules: ${p.rule_version}`, 'hint')));
  out.append(details('Missing information', list(p.missing)));
}

function sourceLink(reference, methodTitle) {
  const label = reference.url.startsWith('https://github.com/') ? 'Code' : reference.url.startsWith('https://distill.pub/') ? 'Article' : 'Paper';
  const a = node('a', `${label} ↗`); a.href = reference.url; a.target = '_blank'; a.rel = 'noopener noreferrer';
  a.setAttribute('aria-label', `${label}: ${methodTitle} — ${reference.title}`);
  return a;
}

function renderMethods(p) {
  const out = $('methods');
  if (!p.recommendations.length) {
    out.append(node('p', 'No suggestions for this model yet.', 'hint'));
    return;
  }
  out.append(node('p', 'Suggestions only; methods not run.', 'hint'));
  for (const m of p.recommendations) {
    const box = node('article', null, 'method');
    const heading = node('div', null, 'method-title'); heading.append(node('h3', m.title));
    if (m.references.length) heading.append(sourceLink(m.references[0], m.title));
    box.append(heading, node('p', m.purpose));
    const expand = details('Details');
    expand.append(node('h4', 'Why'), node('p', m.reason), node('h4', 'Needs'), list(m.prerequisites), node('h4', 'Limits'), list(m.limitations));
    for (const r of m.references.slice(1)) { const div = node('div'); div.append(sourceLink(r, m.title)); expand.append(div); }
    box.append(expand); out.append(box);
  }
}

async function requestInspection(files) {
  if (!files.length || files.length > 3) throw new Error('Select one .pt, one .py and an optional JSON config.');
  if (files.reduce((n, f) => n + f.size, 0) > 128 * 1048576) throw new Error('Combined upload exceeds 128 MiB.');
  const response = await fetch('/api/inspect', {method: 'POST', headers: {'Content-Type': 'application/octet-stream', 'X-MockExplain-Files': encodeURIComponent(JSON.stringify(files.map(f => ({name: f.name, size: f.size}))))}, body: new Blob(files)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Inspection failed. Try a supported model file.');
  return result;
}
async function upload(files) {
  if (!files.length) return;
  try { await store.inspect(files.map(f => f.name).join(' + '), files, requestInspection); }
  catch (error) { message(error.message, 'error'); }
}

// Report imports are data only. Validate fields used by rendering, never evaluate them.
function validateReport(p) {
  const str = x => typeof x === 'string' && x.length <= 20000;
  const strings = x => Array.isArray(x) && x.length <= 50000 && x.every(str);
  const arr = x => Array.isArray(x) && x.length <= 50000;
  const num = x => typeof x === 'number' && Number.isFinite(x) && x >= 0;
  const maybeNum = x => x === null || num(x);
  if (!p || p.schema_version !== 1 || !/^[a-f0-9]{64}$/.test(p.id) || !str(p.reader_version) || !str(p.rule_version) || !arr(p.inputs) || !p.inputs.length || !p.inputs.every(i => str(i.name) && str(i.kind) && num(i.bytes) && /^[a-f0-9]{64}$/.test(i.sha256))) return false;
  const f = p.facts, c = p.coverage, inf = p.inferences;
  if (!f || !c || !inf || !str(inf.family) || !str(inf.confidence) || !str(inf.note) || !arr(inf.evidence) || !inf.evidence.every(e => str(e.source) && str(e.evidence)) || !strings(p.missing) || !strings(f.dtypes) || !strings(f.classes) || !str(f.parameter_count_note) || ![f.tensor_element_count, f.parameter_count, f.storage_bytes].every(maybeNum)) return false;
  if (!arr(f.tensors) || !f.tensors.every(t => str(t.name) && str(t.dtype) && str(t.device_recorded) && num(t.numel) && arr(t.shape) && t.shape.every(num) && arr(t.stride) && t.stride.every(num) && (t.statistics === null || (t.statistics && [t.statistics.sampled_values, t.statistics.total_values, t.statistics.zero_fraction, t.statistics.nonfinite_values].every(num) && [t.statistics.min, t.statistics.max, t.statistics.mean].every(x => x === null || typeof x === 'number' && Number.isFinite(x)))))) return false;
  if (!arr(f.declarations) || !f.declarations.every(d => str(d.name) && str(d.constructor) && str(d.source) && num(d.line)) || (f.config !== null && (!f.config || !str(f.config.note) || typeof f.config.fields !== 'object'))) return false;
  if (!str(c.metadata) || !str(c.partial_reason) || !strings(c.notes) || ![c.tensors_inspected, c.tensors_sampled, c.sampled_values, c.sample_bytes].every(num) || !maybeNum(c.total_tensor_values) || !c.sample_limit || ![c.sample_limit.tensors, c.sample_limit.values_per_tensor, c.sample_limit.bytes].every(num)) return false;
  return arr(p.recommendations) && p.recommendations.every(m => str(m.id) && str(m.title) && str(m.purpose) && str(m.reason) && strings(m.prerequisites) && strings(m.limitations) && arr(m.references) && m.references.every(r => str(r.title) && str(r.url) && /^https:\/\//.test(r.url)));
}

function markdown(p) {
  const clean = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
  const lines = ['# MockExplain model report', '', `Profile: ${p.id}`, `Rules: ${p.rule_version}`, `Reader: ${p.reader_version}`, '', '## Input identity'];
  for (const i of p.inputs) lines.push(`- ${clean(i.name)} — ${i.bytes} bytes — SHA-256 ${i.sha256}`);
  lines.push('', '## Architecture inference', clean(p.inferences.family), clean(p.inferences.note), '', '## Observed facts', '```json', JSON.stringify(p.facts, null, 2).replaceAll('`', '\\u0060'), '```', '', '## Inspection coverage', '```json', JSON.stringify(p.coverage, null, 2), '```', '', '## Structural clues');
  for (const e of p.inferences.evidence) lines.push(`- ${clean(e.source)}: ${clean(e.evidence)}`);
  lines.push('', '## Missing information'); for (const m of p.missing) lines.push(`- ${clean(m)}`);
  lines.push('', '## Suggested methods — not run');
  for (const m of p.recommendations) { lines.push('', `### ${clean(m.title)}`, clean(m.purpose), '', `Why: ${clean(m.reason)}`, '', 'Prerequisites:'); for (const s of m.prerequisites) lines.push(`- ${clean(s)}`); lines.push('', 'Limitations:'); for (const s of m.limitations) lines.push(`- ${clean(s)}`); for (const r of m.references) lines.push(`- [${clean(r.title)}](${r.url})`); }
  return lines.join('\n') + '\n';
}
function download(format) {
  if (store.active?.status !== 'ready') return;
  const p = store.active.profile;
  const blob = new Blob([format === 'json' ? JSON.stringify(p, null, 2) : markdown(p)], {type: format === 'json' ? 'application/json' : 'text/markdown'});
  const url = URL.createObjectURL(blob); const a = node('a'); a.href = url; a.download = `mockexplain-${p.id.slice(0, 12)}.${format}`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function renderComparison() {
  const left = store.profiles.find(e => e.key === $('compare-left').value)?.profile;
  const right = store.profiles.find(e => e.key === $('compare-right').value)?.profile;
  if (!left || !right) return;
  const rows = [
    ['Family · inference', left.inferences.family, right.inferences.family],
    ['Tensor elements', number(left.facts.tensor_element_count), number(right.facts.tensor_element_count)], ['Trainable parameters', number(left.facts.parameter_count), number(right.facts.parameter_count)],
    ['Raw storage', bytes(left.facts.storage_bytes), bytes(right.facts.storage_bytes)], ['Dtypes', left.facts.dtypes.join(', ') || 'Unknown', right.facts.dtypes.join(', ') || 'Unknown'],
    ['Tensor entries', left.facts.tensors.length, right.facts.tensors.length], ['Source call sites', left.facts.declarations.length, right.facts.declarations.length],
    ['Sampling coverage', `${left.coverage.sampled_values} / ${number(left.coverage.total_tensor_values)}`, `${right.coverage.sampled_values} / ${number(right.coverage.total_tensor_values)}`],
    ['Suggested methods', left.recommendations.map(m => m.title).join('; ') || 'None', right.recommendations.map(m => m.title).join('; ') || 'None']];
  const wrap = table(['Observed / inferred field', label(left), label(right)], rows); wrap.classList.add('comparison-table');
  $('comparison-content').replaceChildren();
  if (left.rule_version !== right.rule_version) $('comparison-content').append(node('p', 'Different rule versions may affect suggestions.', 'subtle'));
  $('comparison-content').append(wrap);
}

$('upload').onclick = () => $('files').click();
$('upload-folder').onclick = () => $('folder').click();
$('folder').onchange = event => {
  const all = [...event.target.files]; event.target.value = '';
  const files = all.filter(f => /\.(pt|py|json)$/i.test(f.name));
  if (files.length) upload(files);
  else if (all.length) store.inspect('Selected folder', [], async () => { throw new Error('No .pt or .py files found.'); }).catch(error => message(error.message, 'error'));
};
$('files').onchange = event => { const files = [...event.target.files]; event.target.value = ''; upload(files); };
$('models').onchange = event => store.select(event.target.value);
$('attach').onclick = () => $('attachment').click();
$('attachment').onchange = event => {
  const additions = [...event.target.files]; event.target.value = '';
  const active = store.active;
  if (!active?.files || !additions.length) return;
  const extensions = new Set(additions.map(f => f.name.split('.').at(-1).toLowerCase()));
  const files = [...active.files.filter(f => !extensions.has(f.name.split('.').at(-1).toLowerCase())), ...additions];
  upload(files); // Attachments create an independently identified profile; the old profile stays available.
};
$('open-report').onclick = () => $('report-file').click();
$('report-file').onchange = async event => {
  const files = [...event.target.files]; event.target.value = '';
  // Clear the old view before async reads; imported results never appear under its identity.
  for (const file of files) {
    let entry;
    try {
      entry = store.add(file.name);
      if (file.size > 32 * 1048576) throw new Error('Saved report exceeds the 32 MiB import limit.');
      const p = JSON.parse(await file.text());
      if (!validateReport(p)) throw new Error('Unsupported or malformed report. Open a MockExplain JSON report.');
      entry.profile = p; entry.label = label(p); entry.status = 'ready'; entry.imported = true;
    } catch (error) { if (entry) { entry.status = 'error'; entry.error = error.message; } else message(error.message, 'error'); }
    render();
  }
};
$('export-json').onclick = () => download('json');
$('export-md').onclick = () => download('md');
$('compare').onclick = () => {
  const ready = store.profiles.filter(p => p.status === 'ready');
  if (ready.length < 2) return;
  for (const id of ['compare-left', 'compare-right']) { $(id).replaceChildren(); for (const p of ready) $(id).append(new Option(p.label, p.key)); }
  $('compare-left').value = store.active?.status === 'ready' ? store.active.key : ready[0].key;
  $('compare-right').value = ready.find(p => p.key !== $('compare-left').value).key;
  renderComparison(); $('comparison').showModal();
};
$('compare-left').onchange = renderComparison; $('compare-right').onchange = renderComparison;
$('close-comparison').onclick = () => $('comparison').close();
const remove = node('button', 'Remove profile'); remove.id = 'remove-profile'; remove.onclick = () => { if (store.active) store.remove(store.active.key); }; document.querySelector('.actions').append(remove);
render();
