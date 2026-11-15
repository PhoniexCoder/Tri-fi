import React, { useState } from 'react';
import { clock } from '../../lib/format';
import Icon from '../Icon';

const MAX_ROWS = 60;

/**
 * Card that owns a chart's title, subtitle and its table-view twin, so every
 * value on a chart is also readable without hovering or telling colours apart.
 *
 * table: { times, series: [{ key, label, values }], format }
 */
export default function ChartCard({ title, subtitle, table, children, actions, className = '', id }) {
  const [asTable, setAsTable] = useState(false);
  return (
    <figure className={`card chart-card ${className}`} aria-labelledby={id ? `${id}-title` : undefined}>
      <header className="card-head">
        <div>
          <h3 id={id ? `${id}-title` : undefined} className="card-title">{title}</h3>
          {subtitle && <p className="card-sub">{subtitle}</p>}
        </div>
        <div className="card-actions">
          {actions}
          {table && (
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setAsTable(!asTable)} aria-pressed={asTable}>
              <Icon name={asTable ? 'chart' : 'table'} size={15} />
              {asTable ? 'Chart' : 'Table'}
            </button>
          )}
        </div>
      </header>
      {asTable && table ? <SeriesTable {...table} /> : children}
    </figure>
  );
}

/** Newest-first table of a time series, thinned to at most 60 rows. */
export function SeriesTable({ times, series, format = (v) => String(v) }) {
  const n = times.length;
  const stride = Math.max(1, Math.ceil(n / MAX_ROWS));
  const rows = [];
  for (let i = n - 1; i >= 0; i -= stride) rows.push(i);
  return (
    <div className="table-wrap" tabIndex={0}>
      <table className="data-table">
        <thead>
          <tr>
            <th scope="col">Time</th>
            {series.map((s) => <th key={s.key} scope="col" className="num">{s.label}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((i) => (
            <tr key={i}>
              <td className="num">{clock(times[i])}</td>
              {series.map((s) => <td key={s.key} className="num">{s.values[i] == null ? '—' : format(s.values[i])}</td>)}
            </tr>
          ))}
          {!rows.length && (
            <tr><td colSpan={series.length + 1} className="muted">No data yet</td></tr>
          )}
        </tbody>
      </table>
      {stride > 1 && <p className="table-note">Showing every {stride}th reading, newest first.</p>}
    </div>
  );
}
