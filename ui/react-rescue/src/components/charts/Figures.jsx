import React from 'react';
import { linear, linePath } from './scale';

/**
 * Sparkline: last N values in the de-emphasis ink, the latest point in the
 * accent. Decorative companion to a number that is always printed beside it.
 */
export function Sparkline({ values, width = 96, height = 28, min, max, color = 'var(--accent)' }) {
  const vals = values.filter((v) => v != null);
  if (vals.length < 2) return <svg width={width} height={height} aria-hidden="true" />;
  const lo = min ?? Math.min(...vals);
  const hi = max ?? Math.max(...vals);
  const t = values.map((_, i) => i);
  const x = linear(0, values.length - 1, 2, width - 4);
  const y = linear(lo, hi === lo ? lo + 1 : hi, height - 3, 3);
  const { d } = linePath(t, values, x, y);
  const last = values.length - 1;
  return (
    <svg width={width} height={height} aria-hidden="true" className="spark">
      <path d={d} fill="none" stroke="var(--spark)" strokeWidth="1.5" strokeLinejoin="round" />
      {values[last] != null && <circle cx={x(last)} cy={y(values[last])} r="3" fill={color} />}
    </svg>
  );
}

/**
 * Meter: a ratio against a limit. Track is a lighter step of the fill's ramp;
 * an optional marker shows the threshold. Value text is always printed.
 */
export function Meter({ value, max = 1, marker, tone = 'accent', label, valueText, thin = false }) {
  const f = value == null ? 0 : Math.max(0, Math.min(1, value / max));
  const m = marker == null ? null : Math.max(0, Math.min(1, marker / max));
  return (
    <div className={`meter meter-${tone} ${thin ? 'meter-thin' : ''}`} role="meter" aria-valuemin={0} aria-valuemax={max}
      aria-valuenow={value ?? 0} aria-label={label} aria-valuetext={valueText}>
      <div className="meter-track">
        <div className="meter-fill" style={{ width: `${f * 100}%` }} />
        {m != null && <div className="meter-marker" style={{ left: `${m * 100}%` }} title={`Threshold ${(m * max * 100).toFixed(0)}%`} />}
      </div>
    </div>
  );
}

/** Stat tile: label, value, optional detail line and sparkline. */
export function StatTile({ label, value, detail, icon, trend, tone, title }) {
  return (
    <div className={`tile ${tone ? `tile-${tone}` : ''}`} title={title}>
      <div className="tile-label">{icon}{label}</div>
      <div className="tile-row">
        <div className="tile-value">{value}</div>
        {trend}
      </div>
      {detail && <div className="tile-detail">{detail}</div>}
    </div>
  );
}

/** 0..4 signal bars with the dBm value as accessible text. */
export function SignalBars({ bars, label }) {
  return (
    <span className="bars" role="img" aria-label={label} title={label}>
      {[1, 2, 3, 4].map((b) => <span key={b} className={b <= bars ? 'on' : ''} style={{ height: `${3 + b * 3}px` }} />)}
    </span>
  );
}
