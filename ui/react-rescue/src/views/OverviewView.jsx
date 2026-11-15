import React, { useMemo } from 'react';
import ChartCard from '../components/charts/ChartCard';
import { Meter, Sparkline, StatTile } from '../components/charts/Figures';
import LineChart from '../components/charts/LineChart';
import EventLog from '../components/EventLog';
import FieldMap from '../components/FieldMap';
import Icon from '../components/Icon';
import Readiness from '../components/Readiness';
import StatusHero from '../components/StatusHero';
import { linkState, num, pct } from '../lib/format';
import { nodeColor, nodeRecords } from '../lib/nodes';
import { detectionBands, pick, recent, times } from '../lib/series';

const RANGE = 60;
const pctFmt = (v) => `${Math.round(v * 100)}%`;

export default function OverviewView({ ctx, go }) {
  const { snapshot, hello, history, ready, hero, items, threshold, nodeThreshold } = ctx;
  const nodeIds = hello?.node_ids ?? [];
  const win = useMemo(() => recent(history.samples, RANGE), [history.version]);
  const t = times(win);
  const records = nodeRecords(snapshot, nodeIds);
  const up = records.filter((r) => linkState(r.link).key !== 'down').length;
  const calibrated = records.filter((r) => r.cal?.done).length;
  const rates = records.map((r) => r.link?.rate_hz).filter((v) => v != null);
  const avgRate = rates.length ? rates.reduce((a, b) => a + b, 0) / rates.length : null;
  const rescuer = snapshot?.rescuer;

  const fused = { key: 'c', label: 'Fused confidence', short: 'Fused', color: 'var(--series-7)', values: pick(win, (s) => s.c), area: true };

  return (
    <div className="view-grid overview">
      <StatusHero state={hero} snapshot={snapshot} threshold={threshold} detectedSince={history.detectedSince} />

      <section className="tiles" aria-label="Key numbers">
        <StatTile label="Nodes sending data" icon={<Icon name="wifi" size={15} />}
          value={snapshot ? `${up} of ${nodeIds.length}` : '—'}
          detail={snapshot ? (up === nodeIds.length ? 'All nodes reporting' : `${nodeIds.length - up} silent`) : 'Waiting'}
          tone={!snapshot ? null : up === nodeIds.length ? null : 'warn'}
          trend={<Sparkline values={pick(win, (s) => Object.values(s.nodes).filter((n) => n.rate >= 1).length)} min={0} max={nodeIds.length || 1} />} />
        <StatTile label="Nodes detecting" icon={<Icon name="target" size={15} />}
          value={snapshot ? `${snapshot.n_detecting}` : '—'}
          detail={`Each node alerts above ${pct(nodeThreshold)}`}
          trend={<Sparkline values={pick(win, (s) => s.nDet)} min={0} max={nodeIds.length || 1} />} />
        <StatTile label="People (estimate)" icon={<Icon name="user" size={15} />}
          value={snapshot ? `${snapshot.ai_detected ? snapshot.survivors : 0}` : '—'}
          detail="From the model's class votes" />
        <StatTile label="Average frame rate" icon={<Icon name="pulse" size={15} />}
          value={avgRate == null ? '—' : `${num(avgRate, 0)}/s`}
          detail="20/s or more per node is healthy"
          trend={<Sparkline values={pick(win, (s) => { const v = Object.values(s.nodes).map((n) => n.rate); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; })} min={0} />} />
        <StatTile label="Calibrated" icon={<Icon name="check" size={15} />}
          value={snapshot ? `${calibrated} of ${nodeIds.length}` : '—'}
          detail={calibrated === nodeIds.length && snapshot ? 'Baseline set' : 'Keep the area clear'} />
        <StatTile label="Rescuer badge" icon={<Icon name="badge" size={15} />}
          value={!snapshot ? '—' : rescuer?.confirmed ? 'Confirmed' : rescuer?.active ? 'Online' : 'Not seen'}
          detail={rescuer?.confirmed ? 'Rescue marked on the map' : 'Optional; confirms rescues'} />
      </section>

      <div className="stack">
        <section className="card map-card" aria-labelledby="ov-map-title">
          <header className="card-head">
            <div>
              <h3 id="ov-map-title" className="card-title">Search area</h3>
              <p className="card-sub">Positions are estimates from signal strength; people are shown as areas</p>
            </div>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => go('map')}>Open map</button>
          </header>
          <FieldMap snapshot={snapshot} hello={hello} heatmap={history.heatmap} maxHeight={460} />
        </section>

        <ChartCard id="ov-conf" title="Detection confidence, last minute"
          subtitle="Shaded where an alert was on. Hover or use arrow keys for exact values."
          table={{ times: t, series: [fused], format: pctFmt }}>
          <LineChart times={t} series={[fused]} range={RANGE} yMin={0} yMax={1} format={pctFmt} height={220}
            refs={[{ y: threshold, label: `Alert threshold ${pct(threshold)}`, tone: 'warn' }]}
            bands={detectionBands(win)} label="Fused detection confidence over the last minute" />
        </ChartCard>
      </div>
      <div className="stack">
        <Readiness items={items} ready={ready} />
        <section className="card" aria-labelledby="ov-agree-title">
          <header className="card-head">
            <div>
              <h3 id="ov-agree-title" className="card-title">Node agreement</h3>
              <p className="card-sub">Model probability per node; marker = node threshold {pct(nodeThreshold)}</p>
            </div>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => go('nodes')}>Details</button>
          </header>
          <ul className="meter-list">
            {records.map(({ id, node, link }) => {
              const h = linkState(link);
              return (
                <li key={id}>
                  <span className="ml-name"><span className="swatch" style={{ background: nodeColor(nodeIds, id) }} />Node {id}</span>
                  <Meter value={node?.csi_ml_prob} marker={nodeThreshold} tone={node?.csi_detected ? 'alert' : 'accent'}
                    label={`Node ${id} model probability`} valueText={pct(node?.csi_ml_prob)} />
                  <span className="ml-val">{pct(node?.csi_ml_prob)}</span>
                  <span className={`ml-state tone-text-${node?.csi_detected ? 'alert' : h.tone}`}>
                    {node?.csi_detected ? 'Detecting' : h.label}
                  </span>
                </li>
              );
            })}
            {!records.length && <li className="muted">No nodes configured yet</li>}
          </ul>
        </section>
        <section className="card" aria-labelledby="ov-ev-title">
          <header className="card-head">
            <h3 id="ov-ev-title" className="card-title">Recent events</h3>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => go('events')}>All events</button>
          </header>
          <EventLog events={history.events} limit={6} />
        </section>
      </div>
    </div>
  );
}
