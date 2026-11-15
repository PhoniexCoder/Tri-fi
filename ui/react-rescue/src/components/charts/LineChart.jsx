import React, { useMemo, useState } from 'react';
import { ago, clock } from '../../lib/format';
import { autoDomain, linear, linePath, nearestIndex, niceTicks, useWidth } from './scale';

const M = { top: 12, right: 16, bottom: 26, left: 48 };
const END_LABEL_W = 78;

/**
 * Time-series line chart. Hairline grid, 2px lines, end dot with a surface
 * ring, crosshair + one tooltip listing every series, keyboard stepping,
 * legend for 2+ series (click to hide/show), direct end labels for <= 4
 * series when they don't collide.
 *
 * times:  ascending epoch seconds; series[].values aligned with times (null = gap)
 * range:  seconds shown, ending at `now`
 */
export default function LineChart({
  times, series, now, range, height = 200, yMin, yMax, format = (v) => v.toFixed(0),
  refs = [], bands = [], step = false, label, legend = true, unit = '',
}) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState(null);
  const [hidden, setHidden] = useState(() => new Set());
  const shown = series.filter((s) => !hidden.has(s.key));
  const direct = shown.length <= 4 && width > 420;
  const right = M.right + (direct ? END_LABEL_W : 0);
  const plotW = Math.max(40, width - M.left - right);
  const plotH = height - M.top - M.bottom;
  const t1 = now ?? times[times.length - 1] ?? 0;
  const t0 = t1 - range;

  const x = linear(t0, t1, M.left, M.left + plotW);
  const [d0, d1] = useMemo(
    () => autoDomain([...(shown.length ? shown : series), { values: refs.map((r) => r.y) }], { min: yMin, max: yMax }),
    [shown, series, refs, yMin, yMax],
  );
  const y = linear(d0, d1, M.top + plotH, M.top);
  const yTicks = niceTicks(d0, d1, height < 160 ? 2 : 4).filter((v) => v >= d0 && v <= d1);
  const xTicks = useMemo(() => {
    const stepS = range <= 60 ? 15 : range <= 300 ? 60 : 120;
    const out = [];
    for (let s = 0; s <= range; s += stepS) out.push(s);
    return out;
  }, [range]);

  // Only the visible window; decimate so we never draw more points than pixels.
  const view = useMemo(() => {
    const first = Math.max(0, nearestIndex(times, t0) - 1);
    const idx = [];
    const stride = Math.max(1, Math.floor((times.length - first) / plotW));
    for (let i = first; i < times.length; i += stride) idx.push(i);
    if (idx[idx.length - 1] !== times.length - 1 && times.length) idx.push(times.length - 1);
    return idx;
  }, [times, t0, plotW]);
  const vt = view.map((i) => times[i]);

  const ends = shown.map((s) => {
    for (let i = s.values.length - 1; i >= 0; i--) if (s.values[i] != null) return { s, v: s.values[i], py: y(s.values[i]) };
    return null;
  }).filter(Boolean);
  const sorted = [...ends].sort((a, b) => a.py - b.py);
  const collide = sorted.some((e, i) => i > 0 && e.py - sorted[i - 1].py < 14);

  const hi = hover != null && times[hover] >= t0 ? hover : null;
  const pointerMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * width;
    const t = t0 + ((px - M.left) / plotW) * range;
    setHover(times.length ? nearestIndex(times, Math.min(t1, Math.max(t0, t))) : null);
  };
  const keyDown = (e) => {
    if (!times.length) return;
    const cur = hi ?? times.length - 1;
    if (e.key === 'ArrowLeft') { setHover(Math.max(nearestIndex(times, t0), cur - 1)); e.preventDefault(); }
    if (e.key === 'ArrowRight') { setHover(Math.min(times.length - 1, cur + 1)); e.preventDefault(); }
    if (e.key === 'Escape') setHover(null);
  };
  const toggle = (key) => setHidden((h) => {
    const n = new Set(h);
    if (n.has(key)) n.delete(key);
    else if (n.size < series.length - 1) n.add(key);
    return n;
  });

  const tipLeft = hi != null ? x(times[hi]) : 0;
  return (
    <div className="lc" ref={ref}>
      {legend && series.length > 1 && (
        <ul className="legend" aria-label="Legend">
          {series.map((s) => (
            <li key={s.key}>
              <button type="button" className={`legend-item ${hidden.has(s.key) ? 'off' : ''}`} onClick={() => toggle(s.key)}
                aria-pressed={!hidden.has(s.key)} title={hidden.has(s.key) ? 'Show this line' : 'Hide this line'}>
                <span className="key-line" style={{ background: s.color }} />
                {s.label}
              </button>
            </li>
          ))}
        </ul>
      )}
      <div className="lc-plot">
        <svg width={width} height={height} role="img" aria-label={label} tabIndex={0}
          onPointerMove={pointerMove} onPointerLeave={() => setHover(null)} onKeyDown={keyDown} onBlur={() => setHover(null)}>
          {bands.map((b, i) => {
            const a = Math.max(x(b.t0), M.left);
            const z = Math.min(x(b.t1), M.left + plotW);
            return z > a ? <rect key={i} className="band" x={a} y={M.top} width={z - a} height={plotH} /> : null;
          })}
          {yTicks.map((v) => (
            <g key={v}>
              <line className="grid" x1={M.left} x2={M.left + plotW} y1={y(v)} y2={y(v)} />
              <text className="tick" x={M.left - 8} y={y(v)} dy="0.32em" textAnchor="end">{format(v)}</text>
            </g>
          ))}
          <line className="axis" x1={M.left} x2={M.left + plotW} y1={M.top + plotH} y2={M.top + plotH} />
          {xTicks.map((s) => (
            <text key={s} className="tick" x={x(t1 - s)} y={height - 6}
              textAnchor={s === 0 ? 'end' : s === range ? 'start' : 'middle'}>{ago(s)}</text>
          ))}
          {refs.map((r) => (
            <g key={r.label} className={`ref ref-${r.tone ?? 'neutral'}`}>
              <line x1={M.left} x2={M.left + plotW} y1={y(r.y)} y2={y(r.y)} />
              <text x={M.left + 6} y={y(r.y) - 5}>{r.label}</text>
            </g>
          ))}
          {[...shown].reverse().map((s) => {
            const vals = view.map((i) => s.values[i]);
            const { d } = linePath(vt, vals, x, y, step);
            const area = s.area && d ? `${d}V${M.top + plotH}H${x(vt.find((_, i) => vals[i] != null) ?? t0)}Z` : null;
            return (
              <g key={s.key}>
                {area && <path d={area} fill={s.color} opacity="0.1" />}
                <path d={d} fill="none" stroke={s.color} strokeWidth={s.thin ? 1.5 : 2} strokeLinejoin="round" strokeLinecap="round" />
              </g>
            );
          })}
          {ends.map((e) => (
            <circle key={e.s.key} cx={x(t1)} cy={e.py} r="4" fill={e.s.color} className="ring" />
          ))}
          {direct && !collide && ends.map((e) => (
            <text key={e.s.key} className="end-label" x={M.left + plotW + 10} y={e.py} dy="0.32em">
              {e.s.short ?? e.s.label} {format(e.v)}{unit}
            </text>
          ))}
          {hi != null && (
            <g>
              <line className="crosshair" x1={tipLeft} x2={tipLeft} y1={M.top} y2={M.top + plotH} />
              {shown.map((s) => s.values[hi] != null && (
                <circle key={s.key} cx={tipLeft} cy={y(s.values[hi])} r="4" fill={s.color} className="ring" />
              ))}
            </g>
          )}
          {!times.length && <text className="empty" x={M.left + plotW / 2} y={M.top + plotH / 2} textAnchor="middle">Waiting for data</text>}
        </svg>
        {hi != null && (
          <div className="tooltip" role="status"
            style={{ left: Math.min(Math.max(tipLeft, 90), width - 90), top: 0 }}>
            <div className="tooltip-time">{clock(times[hi])} · {ago(t1 - times[hi])}</div>
            {shown.map((s) => (
              <div className="tooltip-row" key={s.key}>
                <span className="key-line" style={{ background: s.color }} />
                <strong>{s.values[hi] == null ? '—' : `${format(s.values[hi])}${unit}`}</strong>
                <span>{s.label}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
