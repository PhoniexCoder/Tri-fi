import React, { useEffect, useState } from 'react';
import { clock, MODE_INFO } from '../lib/format';
import Icon from './Icon';

/** Connection state in words: live, replay, stale, connecting, offline. */
export function connection(status, hello, snapshot, ageSeconds, staleAfter) {
  if (status === 'offline') return { tone: 'bad', icon: 'wifiOff', label: 'Offline', detail: 'Backend not reachable' };
  if (status === 'connecting') return { tone: 'idle', icon: 'wifi', label: 'Connecting', detail: 'Trying to reach the backend' };
  if (!snapshot) return { tone: 'idle', icon: 'wifi', label: 'Connected', detail: 'Waiting for the first update' };
  if (ageSeconds != null && ageSeconds > staleAfter)
    return { tone: 'warn', icon: 'clock', label: 'Updates stopped', detail: `Last update ${ageSeconds.toFixed(0)} s ago` };
  if (hello?.live === false || snapshot.replay) return { tone: 'info', icon: 'replay', label: 'Replay', detail: 'Playing a recorded session' };
  return { tone: 'ok', icon: 'wifi', label: 'Live', detail: 'Receiving live data' };
}

/** Seconds -> "4:07". */
const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

export default function TopBar({ conn, hello, snapshot, onMode, theme, onTheme, rec, onRecord }) {
  const modes = Object.entries(hello?.op_modes ?? {}).map(([k, v]) => ({ id: Number(k), name: v }));
  const active = snapshot?.active_op_mode;
  const [pending, setPending] = useState(null);
  const replay = hello?.live === false;

  useEffect(() => {
    if (pending != null && active === pending) setPending(null);
  }, [active, pending]);
  useEffect(() => {
    if (pending == null) return undefined;
    const t = setTimeout(() => setPending(null), 4000); // backend never confirmed
    return () => clearTimeout(t);
  }, [pending]);

  const choose = (id) => {
    if (id === active || replay) return;
    if (onMode(id)) setPending(id);
  };

  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true"><Icon name="pulse" size={18} /></span>
        <div className="brand-text">
          <strong>TRI-FI</strong>
          <span>Survivor search console</span>
        </div>
      </div>

      <div className={`conn conn-${conn.tone}`} role="status" aria-live="polite" title={conn.detail}>
        <Icon name={conn.icon} size={16} />
        <span className="conn-label">{conn.label}</span>
        <span className="conn-detail">{snapshot ? clock(snapshot.t) : conn.detail}</span>
      </div>

      <div className="topbar-right">
        {rec && (
          rec.active ? (
            <button type="button" className="btn btn-stop btn-sm rec-btn" onClick={() => onRecord(false)}
              title="Stop recording and save the file">
              <Icon name="stop" size={13} /> Stop <span className="rec-time">{mmss(Math.max(0, (rec.last_t ?? 0) - (rec.first_t ?? rec.last_t ?? 0)))}</span>
            </button>
          ) : (
            <button type="button" className="btn btn-record btn-sm rec-btn" onClick={() => onRecord(true)}
              title="Start saving raw CSI packets for labelling and training">
              <Icon name="rec" size={13} /> Record
            </button>
          )
        )}
        <div className="segmented" role="radiogroup" aria-label="Detection mode">
          {modes.map((m) => {
            const on = m.id === active;
            const info = MODE_INFO[m.id] ?? { short: m.name, long: m.name };
            return (
              <button key={m.id} type="button" role="radio" aria-checked={on}
                className={`seg ${on ? 'on' : ''} ${pending === m.id ? 'pending' : ''}`}
                disabled={replay || !snapshot}
                title={replay ? 'Mode is fixed by the recording during a replay' : `${m.name}: ${info.long}`}
                onClick={() => choose(m.id)}>
                {pending === m.id ? 'Switching…' : info.short}
              </button>
            );
          })}
          {!modes.length && <span className="seg-empty">Mode: —</span>}
        </div>
        <button type="button" className="btn btn-ghost icon-btn" onClick={onTheme}
          aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          title={theme === 'dark' ? 'Light theme (for sunlight)' : 'Dark theme (for night)'}>
          <Icon name={theme === 'dark' ? 'sun' : 'moon'} />
        </button>
      </div>
    </header>
  );
}
