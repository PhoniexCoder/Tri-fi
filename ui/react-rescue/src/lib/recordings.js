/** Seconds -> "4:07" or "1:02:07". */
export function hms(seconds) {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return '—';
  const s = Math.round(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`;
}

/** "4:07", "1:02:07" or "247" -> seconds; null when unreadable. */
export function parseHms(text) {
  const t = String(text ?? '').trim();
  if (!/^\d+(:\d{1,2}){0,2}(\.\d+)?$/.test(t)) return null;
  return t.split(':').reduce((acc, part) => acc * 60 + Number(part), 0);
}

/** Bytes -> "4.2 MB". */
export function bytes(n) {
  if (n == null) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(v < 10 && i ? 1 : 0)} ${units[i]}`;
}

/** "session-20261115-101500.jsonl.gz" -> "15 Nov 2026, 10:15" (local reading of the file's clock). */
export function sessionDate(name) {
  const m = /^session-(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})/.exec(name ?? '');
  if (!m) return name ?? '—';
  const d = new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]);
  return d.toLocaleString([], { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false });
}

export const LABEL_TEXT = { empty: 'Empty', present: 'Person present' };
export const LABEL_COLOR = { empty: 'var(--series-1)', present: 'var(--series-2)' };
export const ACTIVITY_TEXT = { '': '—', still: 'Still', breathing: 'Breathing', moving: 'Moving' };
export const PLACE_TEXT = { '': '—', centre: 'Centre', perimeter: 'Near nodes' };

/** Same rule as six_class() in trifi/labels.py: the custom_dataset/ class for a label, or why none. */
export function sixClass({ label, people, activity, place }) {
  if (label === 'empty') return { cls: 0, why: '' };
  const n = Number(people) || 0;
  if (n > 2) return { cls: null, why: '3 or more people (the six classes stop at 2)' };
  if (!['still', 'breathing', 'moving'].includes(activity)) return { cls: null, why: 'set the activity' };
  const moving = activity === 'moving';
  if (n === 2) return { cls: moving ? 5 : 4, why: '' };
  if (!moving) return { cls: 1, why: '' };
  if (!['centre', 'perimeter'].includes(place)) return { cls: null, why: 'set where the person is moving' };
  return { cls: place === 'centre' ? 2 : 3, why: '' };
}
