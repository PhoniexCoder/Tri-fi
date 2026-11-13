import React from 'react';
import Icon from './Icon';

const MODE_HELP = {
  0: 'AI model on CSI windows (recommended)',
  1: 'Legacy signal-strength heuristic',
  2: 'Average of AI model and legacy heuristic',
};
const MODE_LABEL = { 0: 'AI model', 1: 'RSSI only', 2: 'Blend' };

function Connection({ status, live, ageSeconds }) {
  if (status !== 'online') {
    return (
      <span className="pill pill-bad">
        <Icon name="cross" size={14} /> {status === 'connecting' ? 'Connecting…' : 'Disconnected'}
      </span>
    );
  }
  const stale = ageSeconds != null && ageSeconds > 3;
  return (
    <span className={`pill ${stale ? 'pill-warn' : 'pill-ok'}`} title="Time since the last update">
      <Icon name={stale ? 'warn' : 'pulse'} size={14} />
      {live ? 'Live' : 'Replay'} · {ageSeconds == null ? 'waiting' : stale ? `no update for ${Math.floor(ageSeconds)} s` : 'updating'}
    </span>
  );
}

export default function TopBar({ status, hello, snapshot, ageSeconds, onMode, theme, onTheme }) {
  const live = hello?.live !== false;
  const current = snapshot?.active_op_mode;
  const modes = Object.keys(hello?.op_modes ?? {}).map(Number);
  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">
          <Icon name="pulse" size={20} />
        </span>
        <div>
          <div className="brand-name">TRI-FI</div>
          <div className="brand-sub">WiFi survivor search console</div>
        </div>
      </div>

      <div className="topbar-center">
        <Connection status={status} live={live} ageSeconds={ageSeconds} />
      </div>

      <div className="topbar-right">
        {modes.length > 0 && (
          <div className="segmented" role="radiogroup" aria-label="Detection mode">
            <span className="segmented-label">Detection</span>
            {modes.map((m) => (
              <button
                key={m}
                type="button"
                role="radio"
                aria-checked={current === m}
                title={live ? MODE_HELP[m] : 'Mode is fixed during a replay'}
                disabled={!live || status !== 'online'}
                className={current === m ? 'active' : ''}
                onClick={() => onMode(m)}
              >
                {MODE_LABEL[m] ?? hello.op_modes[m]}
              </button>
            ))}
          </div>
        )}
        <button
          type="button"
          className="icon-button"
          onClick={onTheme}
          title={theme === 'dark' ? 'Switch to light theme (better in sunlight)' : 'Switch to dark theme'}
          aria-label="Toggle colour theme"
        >
          <Icon name={theme === 'dark' ? 'sun' : 'moon'} />
        </button>
      </div>
    </header>
  );
}
