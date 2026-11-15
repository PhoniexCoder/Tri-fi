import React, { useState } from 'react';
import FieldMap, { DEFAULT_LAYERS } from '../components/FieldMap';
import Icon, { Status } from '../components/Icon';
import { consensus, duration, pct } from '../lib/format';

const LAYER_LABELS = { heat: 'Likelihood heatmap', area: 'Inside node area', survivors: 'Possible people', rescuer: 'Rescuer badge', labels: 'Node labels' };
const xy = (x, y) => (x == null ? '—' : `${Number(x).toFixed(2)}, ${Number(y).toFixed(2)}`);

export default function MapView({ ctx }) {
  const { snapshot, hello, history } = ctx;
  const [layers, setLayers] = useState(DEFAULT_LAYERS);
  const survivors = snapshot?.ai_detected ? snapshot.survivors_list ?? [] : [];
  const r = snapshot?.rescuer;
  const heat = history.heatmap;
  const peak = heat?.points.reduce((a, p) => (p.v > a.v ? p : a), { v: -1 });
  const where = consensus(snapshot?.consensus_status);

  return (
    <div className="map-layout">
      <section className="card map-card" aria-labelledby="map-title">
        <header className="card-head">
          <div>
            <h3 id="map-title" className="card-title">Search area</h3>
            <p className="card-sub">Unit grid (0–1 each way) matching the node positions in the kit config</p>
          </div>
        </header>
        <fieldset className="layer-row">
          <legend className="sr-only">Map layers</legend>
          <Icon name="layers" size={16} />
          {Object.entries(LAYER_LABELS).map(([k, label]) => (
            <label key={k} className="check-pill">
              <input type="checkbox" checked={layers[k]} onChange={() => setLayers({ ...layers, [k]: !layers[k] })} />
              <span>{label}</span>
            </label>
          ))}
        </fieldset>
        <FieldMap snapshot={snapshot} hello={hello} heatmap={heat} layers={layers} maxHeight={680} />
      </section>

      <div className="stack">
        <section className="card" aria-labelledby="map-det-title">
          <header className="card-head">
            <h3 id="map-det-title" className="card-title">Detection location</h3>
            {snapshot?.ai_detected ? <Status tone="alert">Detected</Status> : <Status tone="idle">None</Status>}
          </header>
          <dl className="kv-list">
            <div className="kv" title={where.long}><dt>Consensus</dt><dd>{snapshot ? where.short : '—'}</dd></div>
            <div className="kv"><dt>Nodes agreeing</dt><dd>{snapshot?.n_detecting ?? '—'}</dd></div>
            <div className="kv" title="Probability-weighted average of the node positions, smoothed over the last 10 updates">
              <dt>Estimated centre</dt><dd>{xy(snapshot?.pos_x, snapshot?.pos_y)}</dd></div>
            <div className="kv"><dt>Zone flag</dt><dd>{snapshot ? (snapshot.ai_zone ? 'In zone' : 'Clear') : '—'}</dd></div>
          </dl>
          <div className="table-wrap">
            <table className="data-table">
              <thead><tr><th scope="col">Person</th><th scope="col" className="num">Position</th><th scope="col" className="num">Confidence</th></tr></thead>
              <tbody>
                {survivors.map((s, i) => (
                  <tr key={i}><th scope="row">Possible person {i + 1}</th><td className="num">{xy(s.x, s.y)}</td><td className="num">{pct(s.intensity)}</td></tr>
                ))}
                {!survivors.length && <tr><td colSpan={3} className="muted">No one detected right now</td></tr>}
              </tbody>
            </table>
          </div>
          {survivors.length > 1 && <p className="note"><Icon name="info" size={14} /> With more than one person, extra positions are spread around the centre; only the count is measured.</p>}
        </section>

        <section className="card" aria-labelledby="map-badge-title">
          <header className="card-head">
            <h3 id="map-badge-title" className="card-title">Rescuer badge</h3>
            {r?.confirmed ? <Status tone="ok">Confirmed</Status> : r?.active ? <Status tone="ok">Online</Status> : <Status tone="idle">Not seen</Status>}
          </header>
          <dl className="kv-list">
            <div className="kv" title="Approximate, from the nodes' signal strength"><dt>Position</dt><dd>{r?.active ? xy(r.x, r.y) : '—'}</dd></div>
            <div className="kv"><dt>Rescue confirmed at</dt><dd>{xy(r?.conf_x, r?.conf_y)}</dd></div>
          </dl>
        </section>

        <section className="card" aria-labelledby="map-heat-title">
          <header className="card-head">
            <h3 id="map-heat-title" className="card-title">Heatmap</h3>
          </header>
          <dl className="kv-list">
            <div className="kv"><dt>Updated</dt><dd>{heat && snapshot ? `${duration(snapshot.t - heat.t)} ago` : 'Not yet'}</dd></div>
            <div className="kv"><dt>Grid</dt><dd>{heat ? `${Math.round(Math.sqrt(heat.points.length))} × ${Math.round(Math.sqrt(heat.points.length))}` : '—'}</dd></div>
            <div className="kv" title="Highest cell: node probabilities weighted by inverse distance"><dt>Peak</dt><dd>{peak && peak.v >= 0 ? `${pct(peak.v)} at ${xy(peak.x, peak.y)}` : '—'}</dd></div>
          </dl>
          <p className="note">Refreshed every 2 s. It spreads node probabilities by distance, so it shows where to look, not an exact spot.</p>
        </section>
      </div>
    </div>
  );
}
