import React from 'react';
import { linkState, pct, signalBars } from '../lib/format';
import Icon, { TONE_ICON } from './Icon';

function Bars({ rssi }) {
  const n = signalBars(rssi);
  return (
    <span className="bars" title={`${rssi ?? '--'} dBm`} aria-label={`Signal ${n} of 4`}>
      {[1, 2, 3, 4].map((i) => (
        <span key={i} className={i <= n ? 'on' : ''} style={{ height: `${4 + i * 3}px` }} />
      ))}
    </span>
  );
}

function Row({ label, value, help }) {
  return (
    <div className="kv" title={help}>
      <span className="kv-label">{label}</span>
      <span className="kv-value">{value}</span>
    </div>
  );
}

/** One card per receiver node: link health, calibration and what it sees. */
export default function NodeList({ snapshot, nodeIds }) {
  const nodes = snapshot?.nodes ?? [];
  const links = snapshot?.links ?? {};
  const cal = snapshot?.calibration ?? {};
  return (
    <section className="card">
      <div className="card-head">
        <h2>Nodes</h2>
        <span className="card-note">{nodeIds.length} receivers</span>
      </div>
      {nodeIds.length === 0 && <div className="empty">Connect to the backend to see the nodes.</div>}
      <div className="node-list">
        {nodeIds.map((id) => {
          const node = nodes.find((n) => n.node_id === id) ?? {};
          const link = links[id] ?? {};
          const state = linkState(link);
          const c = cal[id] ?? {};
          return (
            <article key={id} className={`node tone-${state.tone}`}>
              <header className="node-head">
                <span className="node-badge">{id}</span>
                <span className="node-title">Node {id}</span>
                <span className={`pill pill-${state.tone}`}>
                  <Icon name={TONE_ICON[state.tone]} size={13} /> {state.label}
                </span>
              </header>

              <div className="node-grid">
                <Row label="Data rate" value={`${(link.rate_hz ?? 0).toFixed(0)} Hz`}
                     help="CSI frames received per second (anchor mode targets 100 Hz)" />
                <Row label="Packet loss" value={pct(link.loss, 1)}
                     help="Share of frames missing, from sequence numbers" />
                <Row label="Signal" value={<><Bars rssi={link.rssi} /> {link.rssi ?? '--'} dBm</>}
                     help="Received signal strength" />
                <Row label="Model" value={
                  <span className={node.csi_detected ? 'text-alert' : ''}>
                    {pct(node.csi_ml_prob)}{node.csi_detected ? ' · detecting' : ''}
                  </span>}
                     help="This node's presence-model output" />
                <Row label="Breathing" value={node.breathing > 0 ? `${node.breathing.toFixed(1)} /min` : '—'}
                     help="Strongest rhythm in the 6–30 /min band (FFT estimate)" />
              </div>

              {c.done ? (
                <div className="node-cal done"><Icon name="check" size={13} /> Calibrated</div>
              ) : (
                <div className="node-cal">
                  <div className="progress"><div style={{ width: pct(c.progress ?? 0) }} /></div>
                  <span>Calibrating {pct(c.progress ?? 0)} — keep area clear</span>
                </div>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}
