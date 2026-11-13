import React from 'react';
import Icon, { TONE_ICON } from './Icon';

export default function Readiness({ items, ready }) {
  return (
    <section className="card">
      <div className="card-head">
        <h2>System readiness</h2>
        <span className={`pill ${ready ? 'pill-ok' : 'pill-warn'}`}>
          <Icon name={ready ? 'check' : 'warn'} size={14} />
          {ready ? 'Ready to search' : 'Not ready'}
        </span>
      </div>
      <ul className="checklist">
        {items.map((it) => (
          <li key={it.key} className={`tone-${it.tone}`}>
            <span className="check-icon">
              <Icon name={it.tone === 'idle' ? 'dot' : TONE_ICON[it.tone]} size={16} />
            </span>
            <span className="check-label">{it.label}</span>
            <span className="check-detail">{it.detail}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
