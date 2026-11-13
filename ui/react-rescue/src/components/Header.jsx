import React from 'react';

const STATUS_TEXT = {
  online: 'LINK STABLE',
  connecting: 'CONNECTING…',
  offline: 'LINK LOST',
};

export default function Header({ status, hello, snapshot, onMode }) {
  const modes = hello?.op_modes ?? {};
  const live = hello?.live !== false;
  const current = snapshot?.active_op_mode;
  return (
    <header id="header">
      <div className="logo">
        <div className="logo-text">
          <h1>TRIFI</h1>
          <span>WiFi CSI survivor search • Field console</span>
        </div>
      </div>
      <div className="header-right">
        {!live && <span className="replay-badge">REPLAY</span>}
        <div className="header-divider" />
        <div className="mode-switcher" title={live ? 'Detection mode' : 'Modes are fixed during replay'}>
          {Object.entries(modes).map(([id, label]) => (
            <button
              key={id}
              type="button"
              disabled={!live || status !== 'online'}
              className={`mode-btn ${current === Number(id) ? 'active' : ''}`}
              onClick={() => onMode(Number(id))}
            >
              {label.split(' ')[0]}
            </button>
          ))}
        </div>
        <div className="header-divider" />
        <span id="mode-badge">{(snapshot?.ai_mode ?? 'WAITING FOR DATA').toUpperCase()}</span>
        <div id="ws-badge" className={`ws-badge ${status === 'online' ? 'online' : ''}`}>
          {STATUS_TEXT[status]}
        </div>
      </div>
    </header>
  );
}
