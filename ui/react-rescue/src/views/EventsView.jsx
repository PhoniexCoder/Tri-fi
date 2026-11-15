import React, { useState } from 'react';
import EventLog, { TONE_LABEL } from '../components/EventLog';
import Icon, { TONE_ICON } from '../components/Icon';
import { clock, csvRow, download } from '../lib/format';

/** Full event log with type filters, text search and CSV export. */
export default function EventsView({ ctx }) {
  const { history } = ctx;
  const [tones, setTones] = useState(() => new Set(Object.keys(TONE_LABEL)));
  const [query, setQuery] = useState('');
  const q = query.trim().toLowerCase();
  const list = history.events.filter((e) => tones.has(e.tone) && (!q || e.text.toLowerCase().includes(q)));
  const counts = history.events.reduce((m, e) => ({ ...m, [e.tone]: (m[e.tone] ?? 0) + 1 }), {});

  const toggle = (tone) => setTones((s) => {
    const n = new Set(s);
    if (n.has(tone)) n.delete(tone);
    else n.add(tone);
    return n;
  });
  const exportCsv = () => {
    const rows = [csvRow(['time_iso', 'time', 'type', 'backend_t', 'message']),
      ...history.events.slice().reverse().map((e) => csvRow([new Date(e.t * 1000).toISOString(), clock(e.t), TONE_LABEL[e.tone], e.st ?? '', e.text]))];
    download(`trifi-events-${new Date().toISOString().slice(0, 19).replace(/:/g, '')}.csv`, rows.join('\n'), 'text/csv');
  };

  return (
    <div className="view-col">
      <div className="filter-row" role="toolbar" aria-label="Event filters">
        {Object.entries(TONE_LABEL).map(([tone, label]) => (
          <button key={tone} type="button" className={`chip chip-${tone} ${tones.has(tone) ? 'on' : ''}`} aria-pressed={tones.has(tone)} onClick={() => toggle(tone)}>
            <Icon name={TONE_ICON[tone]} size={14} /> {label} <span className="chip-count">{counts[tone] ?? 0}</span>
          </button>
        ))}
        <label className="search">
          <Icon name="search" size={15} />
          <span className="sr-only">Search events</span>
          <input type="search" placeholder="Search events" value={query} onChange={(e) => setQuery(e.target.value)} />
        </label>
        <div className="push-right btn-group">
          <button type="button" className="btn btn-ghost" onClick={exportCsv} disabled={!history.events.length}>
            <Icon name="download" size={15} /> Download CSV
          </button>
          <button type="button" className="btn btn-ghost" onClick={history.clearEvents} disabled={!history.events.length}>
            <Icon name="trash" size={15} /> Clear
          </button>
        </div>
      </div>
      <section className="card" aria-labelledby="events-title">
        <header className="card-head">
          <div>
            <h3 id="events-title" className="card-title">Event log</h3>
            <p className="card-sub">{list.length} of {history.events.length} shown, newest first. Times are this screen's clock; kept for this browser session only.</p>
          </div>
        </header>
        <EventLog events={list} empty={history.events.length ? 'No events match these filters.' : 'Nothing has happened yet.'} />
      </section>
    </div>
  );
}
