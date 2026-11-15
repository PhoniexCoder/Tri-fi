import React, { useMemo, useRef, useState } from 'react';
import ChartCard from '../components/charts/ChartCard';
import LineChart from '../components/charts/LineChart';
import Icon from '../components/Icon';
import { download, pct } from '../lib/format';
import { nodeColor } from '../lib/nodes';
import { detectionBands, historyCsv, nodeValues, pick, recent, times } from '../lib/series';

const RANGES = [
  { s: 30, label: '30 s' },
  { s: 60, label: '1 min' },
  { s: 300, label: '5 min' },
  { s: 600, label: '10 min' },
];
const pctFmt = (v) => `${Math.round(v * 100)}%`;
const int = (v) => v.toFixed(0);
const one = (v) => v.toFixed(1);

/**
 * Every time series the backend produces. One filter row scopes all charts;
 * pausing freezes every chart on the same moment so they stay comparable.
 */
export default function SignalsView({ ctx }) {
  const { hello, history, threshold, nodeThreshold } = ctx;
  const nodeIds = hello?.node_ids ?? [];
  const [range, setRange] = useState(60);
  const [paused, setPaused] = useState(false);
  const frozen = useRef(null);

  const live = useMemo(() => recent(history.samples, range), [history.version, range]);
  if (!paused) frozen.current = null;
  else if (!frozen.current) frozen.current = live;
  const win = paused ? frozen.current : live;
  const t = times(win);
  const bands = detectionBands(win);
  const perNode = (key, transform = (v) => v) =>
    nodeIds.map((id) => ({
      key: String(id), label: `Node ${id}`, short: `N${id}`, color: nodeColor(nodeIds, id),
      values: nodeValues(win, id, key).map((v) => (v == null ? null : transform(v))),
    }));

  const fusion = [
    { key: 'c', label: 'Fused confidence', short: 'Fused', color: 'var(--series-7)', values: pick(win, (s) => s.c), area: true },
    { key: 'csi', label: 'AI model (CSI consensus)', short: 'AI', color: 'var(--ink-2)', values: pick(win, (s) => s.csi), thin: true },
    { key: 'tri', label: 'Legacy RSSI detector', short: 'Legacy', color: 'var(--ink-3)', values: pick(win, (s) => s.tri), thin: true },
  ];
  const display = [{ key: 'ai', label: 'Display confidence', short: 'Display', color: 'var(--series-7)', values: pick(win, (s) => s.ai), area: true }];
  const counts = [
    { key: 'nDet', label: 'Nodes detecting', short: 'Detecting', color: 'var(--series-7)', values: pick(win, (s) => s.nDet) },
    { key: 'surv', label: 'People (estimate)', short: 'People', color: 'var(--ink-2)', values: pick(win, (s) => s.surv) },
    { key: 'act', label: 'Legacy active nodes', short: 'Legacy', color: 'var(--ink-3)', values: pick(win, (s) => s.act), thin: true },
  ];

  const charts = [
    { id: 'fusion', title: 'Detection confidence by source', series: fusion, fmt: pctFmt, yMin: 0, yMax: 1, wide: true,
      sub: 'Fused value is what the alert uses; the grey lines are its inputs. Shaded where an alert was on.',
      refs: [{ y: threshold, label: `Alert threshold ${pct(threshold)}`, tone: 'warn' }], bands },
    { id: 'nodeprob', title: 'Model probability per node', series: perNode('ml'), fmt: pctFmt, yMin: 0, yMax: 1,
      sub: 'Average of each node\'s last 5 model windows', refs: [{ y: nodeThreshold, label: `Node threshold ${pct(nodeThreshold)}`, tone: 'warn' }] },
    { id: 'counts', title: 'Detections and people', series: counts, fmt: int, yMin: 0, step: true,
      sub: 'Counts per update. People is a model estimate, not a headcount.' },
    { id: 'rssi', title: 'Signal strength per node', series: perNode('rssi'), fmt: int, unit: ' dBm',
      sub: 'dBm, higher (closer to 0) is stronger. Below −75 is poor.' },
    { id: 'rate', title: 'Frame rate per node', series: perNode('rate'), fmt: int, unit: '/s', yMin: 0,
      sub: 'CSI frames per second over the last 2 s', refs: [{ y: 20, label: 'Healthy 20/s', tone: 'neutral' }] },
    { id: 'loss', title: 'Packet loss per node', series: perNode('loss'), fmt: (v) => `${(v * 100).toFixed(1)}%`, yMin: 0,
      sub: 'Missing frames in the sequence', refs: [{ y: 0.05, label: 'Limit 5%', tone: 'neutral' }] },
    { id: 'bpm', title: 'Breathing rhythm per node (CSI)', series: perNode('bpm'), fmt: one, unit: ' /min', yMin: 0,
      sub: 'Strongest 6–30 per minute rhythm. Indicative only; 0 = none found.' },
    { id: 'motion', title: 'Motion per node (on-chip)', series: perNode('motion'), fmt: (v) => v.toFixed(2), yMin: 0,
      sub: 'Motion energy computed on each node, sent once a second' },
    { id: 'display', title: 'Display confidence', series: display, fmt: pctFmt, yMin: 0, yMax: 1,
      sub: 'Legacy rescaled value (0% below 40% raw) kept for older screens. The fused value above is the honest one.' },
  ];

  return (
    <div className="view-col">
      <div className="filter-row" role="toolbar" aria-label="Chart controls">
        <div className="segmented" role="radiogroup" aria-label="Time range">
          {RANGES.map((r) => (
            <button key={r.s} type="button" role="radio" aria-checked={range === r.s} className={`seg ${range === r.s ? 'on' : ''}`}
              onClick={() => setRange(r.s)}>{r.label}</button>
          ))}
        </div>
        <button type="button" className={`btn ${paused ? 'btn-primary' : 'btn-ghost'}`} onClick={() => setPaused(!paused)} aria-pressed={paused}>
          <Icon name={paused ? 'play' : 'pause'} size={15} /> {paused ? 'Resume' : 'Pause'}
        </button>
        <span className="filter-note">{paused ? 'Paused: charts are frozen, data is still being recorded' : `${win.length} updates in view`}</span>
        <button type="button" className="btn btn-ghost push-right" disabled={!history.samples.length}
          onClick={() => download(`trifi-history-${new Date().toISOString().slice(0, 19).replace(/:/g, '')}.csv`, historyCsv(history.samples, nodeIds), 'text/csv')}>
          <Icon name="download" size={15} /> Download CSV
        </button>
      </div>

      <div className="chart-grid">
        {charts.map((c) => (
          <ChartCard key={c.id} id={`sig-${c.id}`} className={c.wide ? 'span-all' : ''} title={c.title} subtitle={c.sub}
            table={{ times: t, series: c.series, format: (v) => `${c.fmt(v)}${c.unit ?? ''}` }}>
            <LineChart times={t} series={c.series} range={range} yMin={c.yMin} yMax={c.yMax} format={c.fmt} unit={c.unit}
              refs={c.refs} bands={c.bands} step={c.step} height={c.wide ? 240 : 200} label={c.title} />
          </ChartCard>
        ))}
      </div>
    </div>
  );
}
