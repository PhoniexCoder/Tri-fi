import React, { useState } from 'react';
import { Meter } from '../components/charts/Figures';
import Icon, { Status } from '../components/Icon';
import { clock, download, duration, MODE_INFO, num, pct } from '../lib/format';
import { historyCsv } from '../lib/series';

function KV({ label, value, hint }) {
  return <div className="kv" title={hint}><dt>{label}</dt><dd>{value}</dd></div>;
}

/** Connection, model, settings, calibration, legacy fields and the raw data. */
export default function SystemView({ ctx }) {
  const { status, hello, snapshot, conn, socket, history, ageSeconds, threshold, nodeThreshold, now } = ctx;
  const [copied, setCopied] = useState(false);
  const nodeIds = hello?.node_ids ?? [];
  const model = snapshot?.model ?? hello?.model;
  const raw = snapshot ? JSON.stringify({ ...snapshot, heatmap: snapshot.heatmap ? `[${snapshot.heatmap.length} cells]` : undefined }, null, 2) : '';
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(raw);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };
  const stamp = () => new Date().toISOString().slice(0, 19).replace(/:/g, '');

  return (
    <div className="system-grid">
      <section className="card" aria-labelledby="sys-conn">
        <header className="card-head">
          <h3 id="sys-conn" className="card-title">Connection</h3>
          <Status tone={conn.tone === 'idle' ? 'idle' : conn.tone}>{conn.label}</Status>
        </header>
        <dl className="kv-list">
          <KV label="Backend address" value={<code>{socket.endpoint.display}</code>} hint="Set VITE_WS_URL to change it" />
          <KV label="Access token" value={socket.endpoint.hasToken ? 'Provided' : 'None'} hint="Add ?token=… to this page's address if the kit uses one" />
          <KV label="Source" value={hello ? (hello.live ? 'Live sensors' : 'Recorded session (replay)') : '—'} />
          <KV label="Data format" value={hello ? `Version ${hello.schema}` : '—'} />
          <KV label="Updates per second" value={`${num(socket.stats.rate, 1)} of ${hello?.broadcast_hz ?? '—'} expected`} hint="Measured on this screen over the last 5 s" />
          <KV label="Last update" value={ageSeconds == null ? '—' : `${ageSeconds.toFixed(1)} s ago`} />
          <KV label="Connected for" value={socket.stats.connectedAt ? duration((now - socket.stats.connectedAt) / 1000) : '—'} />
          <KV label="Reconnects" value={socket.stats.reconnects} />
          <KV label="Backend clock" value={snapshot ? clock(snapshot.t) : '—'} hint="Time stamped by the backend (recording time during a replay)" />
          <KV label="Screen status" value={status} />
        </dl>
      </section>

      <section className="card" aria-labelledby="sys-model">
        <header className="card-head">
          <h3 id="sys-model" className="card-title">Detection model and modes</h3>
          <Status tone={model?.loaded ? 'ok' : model ? 'warn' : 'idle'}>{model?.loaded ? 'Loaded' : model ? 'Not loaded' : 'Unknown'}</Status>
        </header>
        <dl className="kv-list">
          <KV label="Model classes" value={model?.loaded ? model.classes : '—'} hint="2 = presence model; 6 = older activity model" />
          <KV label="Active mode" value={snapshot?.ai_mode ?? '—'} />
        </dl>
        <ul className="mode-list">
          {Object.entries(hello?.op_modes ?? {}).map(([id, name]) => (
            <li key={id} className={Number(id) === snapshot?.active_op_mode ? 'on' : ''}>
              <strong>{name}</strong>
              {Number(id) === snapshot?.active_op_mode && <Status tone="ok">Active</Status>}
              <p>{MODE_INFO[id]?.long}</p>
            </li>
          ))}
        </ul>
      </section>

      <section className="card" aria-labelledby="sys-cfg">
        <header className="card-head"><h3 id="sys-cfg" className="card-title">Thresholds</h3></header>
        <dl className="kv-list">
          <KV label="Alert threshold" value={pct(threshold)} hint="Fused confidence needed to raise an alert (ui_detect_threshold)" />
          <KV label="Node threshold" value={hello?.node_threshold == null ? '—' : pct(nodeThreshold)} hint="Per-node model probability that counts as detecting (node_detect_threshold)" />
          <KV label="Update rate" value={hello?.broadcast_hz ? `${hello.broadcast_hz} per second` : '—'} />
          <KV label="Nodes configured" value={nodeIds.length ? nodeIds.join(', ') : '—'} />
        </dl>
        <p className="note"><Icon name="info" size={14} /> Change these in the kit's TOML config and restart the backend.</p>
      </section>

      <section className="card" aria-labelledby="sys-cal">
        <header className="card-head"><h3 id="sys-cal" className="card-title">Calibration</h3></header>
        <ul className="meter-list">
          {nodeIds.map((id) => {
            const c = snapshot?.calibration?.[id];
            return (
              <li key={id}>
                <span className="ml-name">Node {id}</span>
                <Meter value={c?.progress} tone={c?.done ? 'ok' : 'accent'} label={`Node ${id} calibration`} valueText={pct(c?.progress)} />
                <span className="ml-val">{pct(c?.progress)}</span>
                <span className="ml-state">{c?.done ? 'Done' : c ? 'Running' : '—'}</span>
              </li>
            );
          })}
        </ul>
        <p className="note">Each node learns the empty-area noise from its first 15 windows. Keep people out of the area until all are done.</p>
      </section>

      <section className="card" aria-labelledby="sys-legacy">
        <header className="card-head"><h3 id="sys-legacy" className="card-title">Legacy fields</h3></header>
        <dl className="kv-list">
          <KV label="Raw RSSI (first node)" value={snapshot ? `${snapshot.raw_rssi} dBm` : '—'} />
          <KV label="Raw motion (first node)" value={num(snapshot?.raw_motion, 3)} />
          <KV label="Raw breathing (first node)" value={snapshot ? `${num(snapshot.raw_br, 1)} /min` : '—'} />
          <KV label="Legacy active nodes" value={snapshot?.active_nodes ?? '—'} />
          <KV label="Depth" value={snapshot ? `${num(snapshot.ai_depth, 1)} (not measured yet)` : '—'} />
          <KV label="Zone" value={snapshot?.ai_zone ?? '—'} />
        </dl>
        <p className="note">Kept for older screens and the legacy detector. Not used for alerts in AI mode.</p>
      </section>

      <section className="card" aria-labelledby="sys-export">
        <header className="card-head"><h3 id="sys-export" className="card-title">Export</h3></header>
        <p className="card-sub">This screen keeps the last 10 minutes ({history.samples.length} updates) and {history.events.length} events. For complete records, run the backend with recording on.</p>
        <div className="btn-group">
          <button type="button" className="btn btn-ghost" disabled={!history.samples.length}
            onClick={() => download(`trifi-history-${stamp()}.csv`, historyCsv(history.samples, nodeIds), 'text/csv')}>
            <Icon name="download" size={15} /> History CSV
          </button>
          <button type="button" className="btn btn-ghost" disabled={!history.samples.length}
            onClick={() => download(`trifi-history-${stamp()}.json`, JSON.stringify({ hello, samples: history.samples }, null, 1), 'application/json')}>
            <Icon name="download" size={15} /> History JSON
          </button>
        </div>
      </section>

      <section className="card span-all" aria-labelledby="sys-raw">
        <header className="card-head">
          <div>
            <h3 id="sys-raw" className="card-title">Latest raw update</h3>
            <p className="card-sub">Exactly what the backend sent (heatmap cells summarised)</p>
          </div>
          <button type="button" className="btn btn-ghost btn-sm" onClick={copy} disabled={!raw}>
            <Icon name={copied ? 'check' : 'copy'} size={15} /> {copied ? 'Copied' : 'Copy'}
          </button>
        </header>
        <details className="raw">
          <summary>Show JSON ({raw ? `${raw.length.toLocaleString()} characters` : 'nothing yet'})</summary>
          <pre tabIndex={0}>{raw || 'No update yet.'}</pre>
        </details>
      </section>
    </div>
  );
}
