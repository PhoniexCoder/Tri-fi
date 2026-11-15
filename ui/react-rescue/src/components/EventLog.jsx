import React from 'react';
import { clock } from '../lib/format';
import Icon, { TONE_ICON } from './Icon';

export const TONE_LABEL = { alert: 'Detection', bad: 'Problem', warn: 'Warning', ok: 'Resolved', info: 'Info' };

/** Newest-first event list. Text is set as text content, never as HTML. */
export default function EventLog({ events, limit, empty = 'Nothing has happened yet.' }) {
  const list = limit ? events.slice(0, limit) : events;
  return (
    <ol className="events" aria-label="Events, newest first">
      {list.map((e) => (
        <li key={e.id} className={`event tone-${e.tone}`}>
          <span className="event-icon" title={TONE_LABEL[e.tone]}><Icon name={TONE_ICON[e.tone]} size={15} /></span>
          <time className="event-time" dateTime={new Date(e.t * 1000).toISOString()}>{clock(e.t)}</time>
          <span className="event-text">{e.text}</span>
        </li>
      ))}
      {!list.length && <li className="event-empty">{empty}</li>}
    </ol>
  );
}
