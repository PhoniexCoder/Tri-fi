import React from 'react';
import { ACTIVITY_TEXT, hms, LABEL_COLOR, LABEL_TEXT } from '../lib/recordings';

/**
 * A recording as one bar, its labelled ranges as coloured blocks (a 2px gap
 * between neighbours). Each block has a tooltip and the label text lives in
 * the table below, so colour never carries the meaning alone.
 */
export default function LabelTimeline({ row, selectedId, onSelect }) {
  const first = row.first_t;
  const duration = row.duration;
  if (first == null || !duration) return <p className="muted">No packets in this recording yet.</p>;
  const pct = (t) => `${Math.max(0, Math.min(100, ((t - first) / duration) * 100))}%`;
  const end = row.last_t;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * duration);

  return (
    <div className="timeline">
      <div className="timeline-track" role="list" aria-label="Labelled ranges along the recording">
        {row.segments.map((s) => {
          const stop = s.end ?? end;
          const width = Math.max(0.4, ((stop - s.start) / duration) * 100);
          return (
            <button key={s.id} type="button" role="listitem"
              className={`timeline-seg ${selectedId === s.id ? 'on' : ''} ${s.end == null ? 'open' : ''}`}
              style={{ left: pct(s.start), width: `${width}%`, background: LABEL_COLOR[s.label] }}
              title={`${LABEL_TEXT[s.label]}, ${s.people} ${s.people === 1 ? 'person' : 'people'}, ${ACTIVITY_TEXT[s.activity]} · ${hms(s.start - first)}–${s.end == null ? 'now' : hms(s.end - first)}${s.setup ? ` · ${s.setup}` : ''}`}
              onClick={() => onSelect?.(s.id)} aria-label={`${LABEL_TEXT[s.label]} from ${hms(s.start - first)}`} />
          );
        })}
      </div>
      <div className="timeline-axis" aria-hidden="true">
        {ticks.map((t) => <span key={t}>{hms(t)}</span>)}
      </div>
    </div>
  );
}
