import test from 'node:test';
import assert from 'node:assert/strict';
import {ProfileStore} from '../mockexplain/static/state.mjs';
const deferred = () => { let resolve, reject; const promise = new Promise((a,b) => { resolve=a; reject=b; }); return {promise, resolve, reject}; };

test('out-of-order completions never replace the active model', async () => {
  const store = new ProfileStore(); const a=deferred(), b=deferred();
  const first = store.inspect('cnn.py', [], () => a.promise);
  const firstKey = store.activeKey;
  const second = store.inspect('transformer.py', [], () => b.promise);
  const secondKey = store.activeKey;
  assert.equal(store.active.profile, null);
  a.resolve({id:'cnn', methods:['grad-cam']}); await first;
  assert.equal(store.activeKey, secondKey);
  assert.equal(store.active.profile, null);
  b.resolve({id:'transformer', methods:['j-lens']}); await second;
  assert.equal(store.active.profile.id, 'transformer');
  store.select(firstKey); assert.equal(store.active.profile.id, 'cnn');
});

test('switching during loading and error preserves each profile state', async () => {
  const store = new ProfileStore(); const first=deferred(), second=deferred();
  const a=store.inspect('good.pt', [], () => first.promise); const key=store.activeKey;
  first.resolve({id:'good'}); await a;
  const b=store.inspect('bad.pt', [], () => second.promise); const badKey=store.activeKey;
  assert.equal(store.active.profile, null);
  store.select(key);
  second.reject(new Error('Unsafe checkpoint')); await b;
  assert.equal(store.active.profile.id, 'good');
  store.select(badKey); assert.equal(store.active.profile, null); assert.equal(store.active.error, 'Unsafe checkpoint');
});

test('removing an inspecting profile ignores its late response', async () => {
  const store = new ProfileStore(); const a=deferred();
  const pending=store.inspect('removed.py', [], () => a.promise);
  store.remove(store.activeKey); a.resolve({id:'late'}); await pending;
  assert.equal(store.active, undefined); assert.equal(store.profiles.length, 0);
});
