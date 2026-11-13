export const pct = (v, digits = 0) => `${((v ?? 0) * 100).toFixed(digits)}%`;

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

/** Link health from packet rate and loss. */
export function linkState(link) {
  const rate = link?.rate_hz ?? 0;
  const loss = link?.loss ?? 0;
  if (rate < 1) return { key: 'down', label: 'No data', tone: 'bad' };
  if (rate < 20 || loss > 0.05) return { key: 'weak', label: 'Weak link', tone: 'warn' };
  return { key: 'good', label: 'Good', tone: 'ok' };
}

/** RSSI (dBm) -> 0..4 signal bars. */
export function signalBars(rssi) {
  if (rssi == null || rssi <= -95) return 0;
  if (rssi > -55) return 4;
  if (rssi > -65) return 3;
  if (rssi > -75) return 2;
  return 1;
}

/** "1 node agrees" / "2 nodes agree". */
export const nodesAgree = (n) => `${n} node${n === 1 ? ' agrees' : 's agree'}`;

const CONSENSUS_TEXT = {
  INSIDE: 'inside node area',
  EDGE: 'near one node',
  'OUT OF RANGE': 'no node detecting',
};
export const consensusText = (status) => CONSENSUS_TEXT[status] ?? '—';
