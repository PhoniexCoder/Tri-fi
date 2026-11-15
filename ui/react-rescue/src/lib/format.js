/** Fraction (0..1) -> "42%". Missing values render as an em dash, never as 0%. */
export function pct(v, digits = 0) {
  if (v == null || Number.isNaN(v)) return '—';
  return `${(v * 100).toFixed(digits)}%`;
}

/** Number with fixed digits, em dash when missing. */
export function num(v, digits = 0, unit = '') {
  if (v == null || Number.isNaN(v)) return '—';
  return `${Number(v).toFixed(digits)}${unit ? ` ${unit}` : ''}`;
}

/** Large counts: 1,284 / 12.9K / 4.2M. */
export function compact(v) {
  if (v == null) return '—';
  if (Math.abs(v) < 10000) return Math.round(v).toLocaleString();
  return new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 }).format(v);
}

/** Epoch seconds -> local HH:MM:SS. */
export function clock(t) {
  if (t == null) return '--:--:--';
  return new Date(t * 1000).toLocaleTimeString([], { hour12: false });
}

/** Seconds -> "45 s", "3 min 05 s", "1 h 02 min". */
export function duration(seconds) {
  if (seconds == null || seconds < 0) return '—';
  const s = Math.floor(seconds);
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, '0')} s`;
  return `${Math.floor(s / 3600)} h ${String(Math.floor((s % 3600) / 60)).padStart(2, '0')} min`;
}

/** Seconds before "now" -> axis label: "now", "−30 s", "−2 min". */
export function ago(seconds) {
  if (seconds < 0.5) return 'now';
  if (seconds < 60) return `−${Math.round(seconds)} s`;
  const m = seconds / 60;
  return `−${Number.isInteger(m) ? m : m.toFixed(1)} min`;
}

/** Link health from packet rate and loss. Tone keys map to the status palette. */
export function linkState(link) {
  const rate = link?.rate_hz ?? 0;
  const loss = link?.loss ?? 0;
  if (!link || rate < 1) return { key: 'down', label: 'No data', tone: 'bad', hint: 'No CSI frames in the last 2 s. Check power and Wi-Fi.' };
  if (rate < 20 || loss > 0.05)
    return { key: 'weak', label: 'Weak link', tone: 'warn', hint: 'Below 20 frames/s or above 5% loss. Move the node closer or clear the line of sight.' };
  return { key: 'good', label: 'Good', tone: 'ok', hint: 'At least 20 frames/s with under 5% loss.' };
}

/** RSSI (dBm) -> 0..4 signal bars and a word. */
export function signal(rssi) {
  if (rssi == null || rssi <= -95) return { bars: 0, word: 'None' };
  if (rssi > -55) return { bars: 4, word: 'Excellent' };
  if (rssi > -65) return { bars: 3, word: 'Good' };
  if (rssi > -75) return { bars: 2, word: 'Fair' };
  return { bars: 1, word: 'Poor' };
}

/** "1 node agrees" / "2 nodes agree". */
export const nodesAgree = (n) => `${n} node${n === 1 ? ' agrees' : 's agree'}`;

const CONSENSUS = {
  INSIDE: { short: 'Inside node area', long: 'Two or more nodes see it, so it is likely between them.' },
  EDGE: { short: 'Near one node', long: 'Only one node sees it, so it is probably close to that node, at the edge of coverage.' },
  'OUT OF RANGE': { short: 'No node detecting', long: 'No node currently sees signs of a person.' },
};
export const consensus = (status) => CONSENSUS[status] ?? { short: '—', long: '' };

/** Short operator-facing names for the backend's mode strings. */
export const MODE_INFO = {
  0: { short: 'AI model', long: 'Neural network on CSI only. Most sensitive; needs a loaded model.' },
  1: { short: 'Legacy RSSI', long: 'Signal strength and motion from the nodes. No model needed; less sensitive.' },
  2: { short: 'Safety blend', long: 'Average of the AI model and legacy RSSI. Fewer false alarms from either alone.' },
};

/** Download text as a file without a server round trip. */
export function download(filename, text, type = 'text/plain') {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** RFC 4180 CSV row. */
export const csvRow = (cells) =>
  cells.map((c) => {
    const s = c == null ? '' : String(c);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  }).join(',');
