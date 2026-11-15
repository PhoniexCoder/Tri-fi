import React from 'react';
import Icon from './Icon';

export const VIEWS = [
  { id: 'overview', label: 'Overview', icon: 'home', hint: 'Search status at a glance' },
  { id: 'map', label: 'Map', icon: 'map', hint: 'Where nodes, people and the rescuer are' },
  { id: 'nodes', label: 'Nodes', icon: 'nodes', hint: 'Every sensor node in detail' },
  { id: 'signals', label: 'Signals', icon: 'chart', hint: 'All readings over time' },
  { id: 'events', label: 'Events', icon: 'list', hint: 'What happened, in order' },
  { id: 'data', label: 'Data', icon: 'database', hint: 'Recordings and labels for model training' },
  { id: 'setup', label: 'Setup', icon: 'usb', hint: 'Install firmware on boards plugged into this laptop' },
  { id: 'system', label: 'System', icon: 'cog', hint: 'Connection, model, settings and raw data' },
];

/** Side navigation (desktop), icon rail (tablet) and bottom bar (phone). Keys 1-8 switch views. */
export default function Nav({ view, onView, badges = {} }) {
  return (
    <nav className="nav" aria-label="Views">
      <ul role="tablist" aria-orientation="vertical">
        {VIEWS.map((v, i) => (
          <li key={v.id}>
            <button type="button" role="tab" aria-selected={view === v.id} aria-controls={`view-${v.id}`}
              className={`nav-item ${view === v.id ? 'on' : ''}`} onClick={() => onView(v.id)}
              title={`${v.hint} (key ${i + 1})`} aria-keyshortcuts={String(i + 1)}>
              <Icon name={v.icon} size={20} />
              <span className="nav-label">{v.label}</span>
              {badges[v.id] && <span className={`nav-badge nav-badge-${badges[v.id].tone}`}>{badges[v.id].text}</span>}
            </button>
          </li>
        ))}
      </ul>
      <p className="nav-foot">Keys 1–8 switch views</p>
    </nav>
  );
}
