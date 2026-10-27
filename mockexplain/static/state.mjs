/** Per-profile request state: switching never reuses another model's result. */
export class ProfileStore {
  constructor(onChange = () => {}) {
    this.profiles = [];
    this.activeKey = null;
    this.onChange = onChange;
    this.sequence = 0;
  }
  get active() { return this.profiles.find(p => p.key === this.activeKey); }
  select(key) {
    if (!this.profiles.some(p => p.key === key)) return;
    this.activeKey = key;
    this.onChange();
  }
  add(label, files = null) {
    if (this.profiles.length >= 12) throw new Error('12 profiles are open. Save or remove one first.');
    const entry = {key: String(++this.sequence), label, files, status: 'loading', profile: null, error: null};
    this.profiles.push(entry);
    this.activeKey = entry.key;
    this.onChange();
    return entry;
  }
  async inspect(label, files, request) {
    const entry = this.add(label, files);
    try {
      const profile = await request(files);
      if (!this.profiles.includes(entry)) return;
      entry.profile = profile;
      entry.status = 'ready';
    } catch (error) {
      if (!this.profiles.includes(entry)) return;
      entry.error = error.message || 'Inspection failed. Try another supported file.';
      entry.status = 'error';
    }
    this.onChange();
    return entry;
  }
  import(profile, label) {
    const entry = this.add(label);
    entry.profile = profile;
    entry.status = 'ready';
    entry.imported = true;
    this.onChange();
  }
  remove(key) {
    this.profiles = this.profiles.filter(p => p.key !== key);
    if (this.activeKey === key) this.activeKey = this.profiles.at(-1)?.key || null;
    this.onChange();
  }
}
