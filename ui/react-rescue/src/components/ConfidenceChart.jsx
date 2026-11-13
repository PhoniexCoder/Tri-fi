import React from 'react';
import { pct } from '../lib/format';

const W = 320;
const H = 110;
const PAD = { l: 34, r: 6, t: 6, b: 18 };
const SPAN = 60;

/** Last 60 s of model confidence, with the alert threshold and detection periods. */
export default function ConfidenceChart({ points, threshold = 0.3 }) {
  const end = points.length ? points[points.length - 1].t : 0;
  const x = (t) => PAD.l + ((t - (end - SPAN)) / SPAN) * (W - PAD.l - PAD.r);
  const y = (v) => PAD.t + (1 - v) * (H - PAD.t - PAD.b);

  const line = points.map((p, i) => `${i ? 'L' : 'M'}${x(p.t).toFixed(1)} ${y(p.c).toFixed(1)}`).join(' ');
  const area = points.length
    ? `${line} L${x(end).toFixed(1)} ${y(0)} L${x(points[0].t).toFixed(1)} ${y(0)} Z`
    : '';

  // Contiguous detection periods as shaded bands.
  const bands = [];
  let start = null;
  points.forEach((p, i) => {
    if (p.d && start == null) start = p.t;
    if ((!p.d || i === points.length - 1) && start != null) {
      bands.push([start, p.d ? p.t : points[i - 1]?.t ?? p.t]);
      start = null;
    }
  });

  return (
    <section className="card">
      <div className="card-head">
        <h2>Confidence, last 60 s</h2>
        <span className="legend-inline">
          <span className="swatch swatch-band" /> detection
        </span>
      </div>
      {points.length < 2 ? (
        <div className="empty">Waiting for data…</div>
      ) : (
        <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img"
             aria-label={`Confidence over the last minute, now ${pct(points[points.length - 1].c)}`}>
          {bands.map(([a, b]) => (
            <rect key={a} className="chart-band" x={x(a)} y={PAD.t} width={Math.max(2, x(b) - x(a))}
                  height={H - PAD.t - PAD.b} />
          ))}
          {[0, 0.5, 1].map((v) => (
            <g key={v}>
              <line className="chart-grid" x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} />
              <text className="chart-axis" x={PAD.l - 6} y={y(v) + 4} textAnchor="end">{pct(v)}</text>
            </g>
          ))}
          <line className="chart-threshold" x1={PAD.l} x2={W - PAD.r} y1={y(threshold)} y2={y(threshold)} />
          <path className="chart-area" d={area} />
          <path className="chart-line" d={line} />
          <text className="chart-axis" x={PAD.l} y={H - 4}>−60 s</text>
          <text className="chart-axis" x={W - PAD.r} y={H - 4} textAnchor="end">now</text>
        </svg>
      )}
    </section>
  );
}
