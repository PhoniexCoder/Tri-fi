import React from 'react';
import { clock } from '../lib/format';
import Icon, { TONE_ICON } from './Icon';

/** Newest-first log of what changed: detections, links, calibration, mode. */
export default function EventLog({ events }) {
  return (
    <section className="card">
      <div className="card-head">
        <h2>Event log</h2>
        <span className="card-note">newest first</span>
      </div>
      {events.length === 0 ? (
        <div className="empty">Nothing yet.</div>
      ) : (
        <ol className="events">
          {events.map((e) => (
            <li key={e.id} className={`tone-${e.tone}`}>
              <time>{clock(e.t)}</time>
              <Icon name={TONE_ICON[e.tone] ?? 'info'} size={15} />
              <span>{e.text}</span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
