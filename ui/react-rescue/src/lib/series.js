import { csvRow } from './format';

/** Samples within the last `range` seconds of the newest one. */
export function recent(samples, range) {
  if (!samples.length) return [];
  const end = samples[samples.length - 1].t;
  let i = samples.length - 1;
  while (i > 0 && samples[i - 1].t >= end - range) i--;
  return samples.slice(i);
}

export const times = (samples) => samples.map((s) => s.t);
export const pick = (samples, fn) => samples.map((s) => { const v = fn(s); return v == null ? null : v; });
export const nodeValues = (samples, id, key) => samples.map((s) => s.nodes[id]?.[key] ?? null);

/** Contiguous runs where a detection was on, as {t0, t1} bands. */
export function detectionBands(samples) {
  const bands = [];
  let start = null;
  for (let i = 0; i < samples.length; i++) {
    if (samples[i].det && start == null) start = samples[i].t;
    if (!samples[i].det && start != null) { bands.push({ t0: start, t1: samples[i].t }); start = null; }
  }
  if (start != null) bands.push({ t0: start, t1: samples[samples.length - 1].t });
  return bands;
}

/** Full history as CSV: one row per snapshot, one column group per node. */
export function historyCsv(samples, nodeIds) {
  const nodeCols = ['model_prob', 'detecting', 'rssi_dbm', 'frames_per_s', 'loss', 'frames', 'skipped_runs', 'motion', 'breathing_per_min', 'breathing_seen'];
  const head = ['time_iso', 't', 'confidence', 'display_confidence', 'csi_prob', 'legacy_prob', 'detected', 'nodes_detecting', 'people_estimate', 'legacy_active_nodes', 'mode',
    ...nodeIds.flatMap((id) => nodeCols.map((c) => `node${id}_${c}`))];
  const rows = samples.map((s) => csvRow([
    new Date(s.t * 1000).toISOString(), s.t, s.c, s.ai, s.csi, s.tri, s.det, s.nDet, s.surv, s.act, s.mode,
    ...nodeIds.flatMap((id) => {
      const n = s.nodes[id] ?? {};
      return [n.ml, n.det, n.rssi, n.rate, n.loss, n.frames, n.drop, n.motion, n.bpm, n.bdet];
    }),
  ]));
  return [csvRow(head), ...rows].join('\n');
}
