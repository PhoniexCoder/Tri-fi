import { useEffect, useRef, useState } from 'react';
import { linkState, nodesAgree, pct } from './format';

const HISTORY_SECONDS = 60;
const MAX_EVENTS = 50;

/**
 * Keeps the last 60 s of confidence and derives a plain-language event log by
 * comparing each snapshot with the previous one.
 */
export function useHistory(snapshot, status, hello) {
  const [points, setPoints] = useState([]);
  const [events, setEvents] = useState([]);
  const [detectedSince, setDetectedSince] = useState(null);
  const prev = useRef({ snapshot: null, status: null, links: {} });

  // Every entry is stamped with the time the dashboard saw it (one clock for the whole log).
  const log = (_t, tone, text) => {
    const t = Date.now() / 1000;
    setEvents((list) => [{ id: `${t}-${text}`, t, tone, text }, ...list].slice(0, MAX_EVENTS));
  };

  // Connection changes (wall-clock time).
  useEffect(() => {
    const before = prev.current.status;
    const now = Date.now() / 1000;
    if (before && before !== status) {
      if (status === 'online') log(now, 'ok', 'Connected to the backend');
      if (status === 'offline' && before === 'online') log(now, 'bad', 'Connection to the backend lost');
    }
    prev.current.status = status;
  }, [status]);

  // A new session (fresh hello) starts a fresh history.
  useEffect(() => {
    setPoints([]);
    setDetectedSince(null);
    prev.current.snapshot = null;
    prev.current.links = {};
  }, [hello]);

  useEffect(() => {
    if (!snapshot) return;
    const t = snapshot.t;
    const last = prev.current.snapshot;

    setPoints((list) => {
      const next = [...list, { t, c: snapshot.confidence_raw ?? 0, d: snapshot.ai_detected }];
      return next.filter((p) => p.t >= t - HISTORY_SECONDS);
    });

    if (!last) {
      log(t, 'info', `Receiving data${snapshot.replay ? ' (replay)' : ''}`);
      if (snapshot.ai_detected) setDetectedSince(t);
    } else {
      if (snapshot.ai_detected && !last.ai_detected) {
        setDetectedSince(t);
        log(t, 'alert', `Possible survivor detected (${pct(snapshot.confidence_raw)}, ${nodesAgree(snapshot.n_detecting)})`);
      }
      if (!snapshot.ai_detected && last.ai_detected) {
        setDetectedSince(null);
        log(t, 'info', 'Detection cleared');
      }
      if (snapshot.active_op_mode !== last.active_op_mode) {
        log(t, 'info', `Detection mode changed to ${snapshot.ai_mode}`);
      }
      if (snapshot.rescuer?.confirmed && !last.rescuer?.confirmed) {
        log(t, 'ok', 'Rescuer badge confirmed a rescue');
      }
      for (const [id, cal] of Object.entries(snapshot.calibration ?? {})) {
        if (cal.done && !last.calibration?.[id]?.done) log(t, 'ok', `Node ${id} calibrated`);
      }
    }

    for (const [id, link] of Object.entries(snapshot.links ?? {})) {
      const state = linkState(link).key;
      const before = prev.current.links[id];
      if (before && before !== state) {
        if (state === 'down') log(t, 'bad', `Node ${id} stopped sending data`);
        else if (state === 'weak') log(t, 'warn', `Node ${id} link is weak (${link.rate_hz.toFixed(0)} Hz, ${pct(link.loss, 1)} loss)`);
        else log(t, 'ok', `Node ${id} link is good again`);
      }
      prev.current.links[id] = state;
    }

    prev.current.snapshot = snapshot;
  }, [snapshot]);

  return { points, events, detectedSince };
}
