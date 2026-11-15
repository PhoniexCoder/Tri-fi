import React from 'react';
import { compact, linkState, num, pct, signal } from '../lib/format';
import { Meter, SignalBars, Sparkline } from './charts/Figures';
import { Status } from './Icon';

/** Label + value row with an explanation on hover. */
function Row({ label, value, hint, children }) {
  return (
    <div className="kv" title={hint}>
      <dt>{label}</dt>
      <dd>{value}{children}</dd>
    </div>
  );
}

/**
 * Everything the backend reports about one node: link, signal, calibration,
 * model output, on-chip vitals, CSI breathing estimate and position.
 */
export default function NodeCard({ id, color, rec, threshold, spark, position }) {
  const { node, link, cal } = rec;
  const health = linkState(link);
  const sig = signal(link?.rssi ?? node?.rssi);
  const detecting = node?.csi_detected;
  return (
    <article className={`card node-card ${detecting ? 'is-detecting' : ''}`} aria-labelledby={`node-${id}-title`}>
      <header className="node-head">
        <span className="node-chip" style={{ background: color }} aria-hidden="true">{id}</span>
        <h3 id={`node-${id}-title`} className="card-title">Node {id}</h3>
        <Status tone={health.tone} title={health.hint}>{health.label}</Status>
      </header>

      <section className="node-section">
        <div className="node-section-head">
          <h4>Detection model</h4>
          {detecting ? <Status tone="alert">Detecting</Status> : <Status tone="idle">Not detecting</Status>}
        </div>
        <div className="node-metric">
          <span className="node-big">{pct(node?.csi_ml_prob)}</span>
          {spark && <Sparkline values={spark.ml} min={0} max={1} color={color} />}
        </div>
        <Meter value={node?.csi_ml_prob} marker={threshold} tone={detecting ? 'alert' : 'accent'}
          label={`Node ${id} model probability`} valueText={`${pct(node?.csi_ml_prob)}, threshold ${pct(threshold)}`} />
        <p className="node-foot">Node detects above {pct(threshold)} (average of last 5 windows)</p>
      </section>

      <dl className="kv-list">
        <Row label="Frames per second" value={num(link?.rate_hz, 1)} hint="CSI frames received in the last 2 s. 20 or more is healthy.">
          {spark && <Sparkline values={spark.rate} min={0} width={64} height={20} color={color} />}
        </Row>
        <Row label="Packet loss" value={pct(link?.loss, 1)} hint="Share of frames missing from the sequence. Under 5% is healthy." />
        <Row label="Signal" value={`${link?.rssi ?? node?.rssi ?? '—'} dBm`} hint={`Received signal strength: ${sig.word}`}>
          <SignalBars bars={sig.bars} label={`${sig.word} signal`} />
        </Row>
        <Row label="Frames received" value={compact(link?.frames)} hint="Total CSI frames decoded from this node since the backend started." />
        <Row label="Skipped model runs" value={compact(link?.dropped_inference)}
          hint="Windows skipped because the previous model run was still busy. A rising number means the laptop is overloaded." />
        <Row label="Calibration" value={cal?.done ? 'Complete' : cal ? `${pct(cal.progress)}` : '—'}
          hint="Empty-area noise baseline (15 windows). Keep people away until it completes." />
        <Row label="Breathing rhythm (CSI)" value={node?.breathing ? `${num(node.breathing, 1)} /min` : '—'}
          hint="Breathing-like rhythm (6–30 per minute) found in the last 30 s of CSI; blank when none is found. Indicative only.">
          {node?.breathing_detected && <span className="tag tag-warn">seen</span>}
        </Row>
        <Row label="Motion (on-chip)" value={num(node?.motion, 2)} hint="Motion energy computed on the node itself, sent once a second." />
        <Row label="Legacy RSSI detector" value={node?.active ? 'Active' : 'Inactive'} hint="Whether the signal-strength detector counts this node as reporting." />
        <Row label="Position" value={position ? `${position[0].toFixed(2)}, ${position[1].toFixed(2)}` : '—'}
          hint="Configured position on the map (0–1 on each axis)." />
      </dl>
    </article>
  );
}
