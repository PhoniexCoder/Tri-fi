import { useEffect, useRef, useState } from 'react';
import { linkState, nodesAgree, pct } from './format';

export const HISTORY_SECONDS = 600; // longest range the Signals view offers
const MAX_EVENTS = 500;

/** Compact per-snapshot record kept for charts and export. */
function toSample(s) {
  const nodes = {};
  for (const n of s.nodes ?? []) {
    const link = s.links?.[n.node_id] ?? {};
    nodes[n.node_id] = {
      ml: n.csi_ml_prob,
      det: n.csi_detected,
      rssi: link.rssi ?? n.rssi,
      rate: link.rate_hz ?? 0,
      loss: link.loss ?? 0,
      frames: link.frames ?? 0,
      drop: link.dropped_inference ?? 0,
      motion: n.motion,
      bpm: n.breathing,
      bdet: n.breathing_detected,
    };
  }
  return {
    t: s.t,
    c: s.confidence_raw ?? 0,
    ai: s.ai_prob ?? 0,
    csi: s.csi_ml_prob ?? 0,
    tri: s.tri_prob ?? 0,
    det: Boolean(s.ai_detected),
    nDet: s.n_detecting ?? 0,
    surv: s.survivors ?? 0,
    act: s.active_nodes ?? 0,
    mode: s.active_op_mode,
    nodes,
  };
}

/**
 * Keeps up to 10 minutes of snapshots (a ref plus a version counter, so 5 Hz
 * updates don't copy the whole buffer) and derives a plain-language event log
 * by comparing each snapshot with the previous one.
 */
export function useHistory({ snapshot, status, hello, error }) {
  const samples = useRef([]);
  const [version, setVersion] = useState(0);
  const [events, setEvents] = useState([]);
  const [detectedSince, setDetectedSince] = useState(null);
  const [heatmap, setHeatmap] = useState(null);
  const prev = useRef({ snapshot: null, status: null, links: {} });

  // Every entry carries the wall-clock time the dashboard saw it, plus the
  // backend's own time (differs during replays).
  const log = (tone, text, st = null) => {
    const t = Date.now() / 1000;
    setEvents((list) => [{ id: `${t}-${Math.random()}`, t, st, tone, text }, ...list].slice(0, MAX_EVENTS));
  };

  useEffect(() => {
    const before = prev.current.status;
    if (before && before !== status) {
      if (status === 'online') log('ok', 'Connected to the backend');
      if (status === 'offline' && before === 'online') log('bad', 'Connection to the backend lost');
    }
    prev.current.status = status;
  }, [status]);

  useEffect(() => {
    if (error) log('bad', `Backend error: ${error}`);
  }, [error]);

  // A new session (fresh hello) starts a fresh history.
  useEffect(() => {
    samples.current = [];
    setVersion((v) => v + 1);
    setDetectedSince(null);
    setHeatmap(null);
    prev.current.snapshot = null;
    prev.current.links = {};
  }, [hello]);

  useEffect(() => {
    if (!snapshot) return;
    const t = snapshot.t;
    const last = prev.current.snapshot;

    const buf = samples.current;
    if (buf.length && t < buf[buf.length - 1].t) buf.length = 0; // time went backwards: new replay
    buf.push(toSample(snapshot));
    while (buf.length && buf[0].t < t - HISTORY_SECONDS) buf.shift();
    setVersion((v) => v + 1);

    if (snapshot.heatmap) setHeatmap({ points: snapshot.heatmap, t });

    if (!last) {
      log('info', `Receiving data${snapshot.replay ? ' from a recording' : ''}`, t);
      if (snapshot.ai_detected) setDetectedSince(t);
    } else {
      if (snapshot.ai_detected && !last.ai_detected) {
        setDetectedSince(t);
        log('alert', `Possible survivor detected: ${pct(snapshot.confidence_raw)} confidence, ${nodesAgree(snapshot.n_detecting)}`, t);
      }
      if (!snapshot.ai_detected && last.ai_detected) {
        setDetectedSince(null);
        log('info', 'Detection cleared', t);
      }
      if (snapshot.ai_detected && last.ai_detected && snapshot.survivors !== last.survivors) {
        log('alert', `Estimated people changed from ${last.survivors} to ${snapshot.survivors}`, t);
      }
      if (snapshot.active_op_mode !== last.active_op_mode) log('info', `Detection mode changed to ${snapshot.ai_mode}`, t);
      if (snapshot.rescuer?.active && !last.rescuer?.active) log('ok', 'Rescuer badge came online', t);
      if (!snapshot.rescuer?.active && last.rescuer?.active) log('warn', 'Rescuer badge went quiet', t);
      if (snapshot.rescuer?.confirmed && !last.rescuer?.confirmed) log('ok', 'Rescuer badge confirmed a rescue', t);
      for (const [id, cal] of Object.entries(snapshot.calibration ?? {})) {
        if (cal.done && !last.calibration?.[id]?.done) log('ok', `Node ${id} calibrated`, t);
      }
      const lastNodes = Object.fromEntries((last.nodes ?? []).map((n) => [n.node_id, n]));
      for (const n of snapshot.nodes ?? []) {
        const b = lastNodes[n.node_id];
        if (!b) continue;
        if (n.csi_detected && !b.csi_detected) log('warn', `Node ${n.node_id} started detecting (${pct(n.csi_ml_prob)})`, t);
        if (!n.csi_detected && b.csi_detected) log('info', `Node ${n.node_id} stopped detecting`, t);
        if (n.breathing_detected && !b.breathing_detected) log('warn', `Node ${n.node_id} sees a breathing-like rhythm (${n.breathing.toFixed(1)} per min)`, t);
      }
    }

    for (const [id, link] of Object.entries(snapshot.links ?? {})) {
      const state = linkState(link).key;
      const before = prev.current.links[id];
      if (before && before !== state) {
        if (state === 'down') log('bad', `Node ${id} stopped sending data`, t);
        else if (state === 'weak') log('warn', `Node ${id} link is weak (${link.rate_hz.toFixed(0)} frames/s, ${pct(link.loss, 1)} loss)`, t);
        else log('ok', `Node ${id} link is good again`, t);
      }
      prev.current.links[id] = state;
    }

    prev.current.snapshot = snapshot;
  }, [snapshot]);

  return { samples: samples.current, version, events, detectedSince, heatmap, clearEvents: () => setEvents([]), addEvent: log };
}
