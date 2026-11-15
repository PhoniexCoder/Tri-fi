import React from 'react';
import { Meter } from './charts/Figures';
import Icon, { TONE_ICON } from './Icon';

/** Pre-search checklist. Each row: icon + label + plain detail, never colour alone. */
export default function Readiness({ items, ready }) {
  return (
    <section className="card" aria-labelledby="ready-title">
      <header className="card-head">
        <div>
          <h3 id="ready-title" className="card-title">Readiness</h3>
          <p className="card-sub">{ready ? 'Ready to search' : 'Fix the marked items before trusting results'}</p>
        </div>
        <span className={`status status-${ready ? 'ok' : 'warn'}`}>
          <Icon name={ready ? 'check' : 'warn'} size={14} />
          <span>{ready ? 'Ready' : 'Not ready'}</span>
        </span>
      </header>
      <ul className="checklist">
        {items.map((it) => (
          <li key={it.key} className={`check-row tone-${it.tone}`}>
            <span className="check-icon"><Icon name={TONE_ICON[it.tone]} size={16} /></span>
            <div className="check-body">
              <span className="check-label">{it.label}</span>
              <span className="check-detail">{it.detail}</span>
              {it.progress != null && <Meter value={it.progress} thin label={`${it.label} progress`} />}
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
