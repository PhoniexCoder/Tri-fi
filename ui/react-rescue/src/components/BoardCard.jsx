import React from 'react';
import { Meter } from './charts/Figures';
import Icon, { Status } from './Icon';

/** One plain status for a board, from its probe and install job. */
export function boardStatus(port) {
  const { probe, job } = port;
  if (job.state === 'running') return { tone: 'info', label: `Installing ${Math.round(job.percent ?? 0)}%` };
  if (job.state === 'done') return { tone: 'ok', label: 'Installed' };
  if (job.state === 'error') return { tone: 'bad', label: 'Failed' };
  if (probe.state === 'probing') return { tone: 'idle', label: 'Checking…' };
  if (probe.state === 'queued') return { tone: 'idle', label: 'Waiting to check' };
  if (probe.state === 'ok') return { tone: 'ok', label: 'Ready' };
  if (probe.state === 'error') return { tone: 'warn', label: 'Needs attention' };
  return { tone: 'idle', label: 'Detected' };
}

function firmwareText(probe, image) {
  if (probe.state !== 'ok') return '—';
  const fw = probe.firmware;
  if (!fw) return 'No TRI-FI firmware found';
  const same = image && fw.project === image.project && fw.version === image.version;
  return `${fw.project} ${fw.version}${fw.date ? `, built ${fw.date}` : ''}${same ? ' (same as install files)' : ''}`;
}

/**
 * A board plugged into the laptop: what it is, how it will be set up
 * (role, node ID, full install or settings only) and the install progress.
 */
export default function BoardCard({ port, cfg, onChange, onInstall, onRetryProbe, blocker, image, configuredIds, snapshot }) {
  const { probe, job } = port;
  const status = boardStatus(port);
  const running = job.state === 'running';
  const doneId = job.state === 'done' ? job.node_id : null;
  const link = doneId != null ? snapshot?.links?.[doneId] : null;
  const joined = (link?.rate_hz ?? 0) >= 1;
  const unknownId = cfg.role === 'receiver' && configuredIds.length > 0 && !configuredIds.includes(cfg.nodeId);
  const flash = probe.flash_mb ? `${probe.flash_mb} MB` : '—';

  return (
    <article className={`card board ${running ? 'is-running' : ''}`} aria-labelledby={`board-${port.device}`}>
      <header className="node-head">
        <span className="board-icon" aria-hidden="true"><Icon name="usb" size={18} /></span>
        <div className="board-title">
          <h3 id={`board-${port.device}`} className="card-title">{port.device}</h3>
          <span className="card-sub">{port.adapter}</span>
        </div>
        <Status tone={status.tone}>{status.label}</Status>
      </header>

      <dl className="kv-list">
        <div className="kv"><dt>Chip</dt><dd>{probe.chip ?? (probe.state === 'probing' ? 'Checking…' : probe.state === 'queued' ? 'Next in line…' : '—')}</dd></div>
        <div className="kv"><dt>Flash</dt><dd>{flash}{probe.flash_mb ? (probe.flash_mb >= 8 ? ' · 8 MB image' : ' · 4 MB image') : ''}</dd></div>
        <div className="kv"><dt>MAC address</dt><dd>{probe.mac ? <code>{probe.mac}</code> : '—'}</dd></div>
        <div className="kv"><dt>Firmware on board</dt><dd className="wrap">{firmwareText(probe, image)}</dd></div>
      </dl>

      {probe.state === 'error' && (
        <div className="inline-alert inline-warn" role="alert">
          <Icon name="warn" size={16} />
          <span>{probe.error}</span>
          <button type="button" className="btn btn-ghost btn-sm" onClick={onRetryProbe} disabled={Boolean(blocker && blocker.busy)}>Check again</button>
        </div>
      )}

      <fieldset className="board-form" disabled={running}>
        <legend className="sr-only">Install options for {port.device}</legend>
        <div className="field">
          <span className="field-label">Role</span>
          <div className="segmented" role="radiogroup" aria-label="Role">
            {[['receiver', 'Receiver'], ['anchor', 'Anchor']].map(([v, label]) => (
              <button key={v} type="button" role="radio" aria-checked={cfg.role === v} className={`seg ${cfg.role === v ? 'on' : ''}`}
                onClick={() => onChange({ role: v, nodeId: v === 'anchor' ? 0 : cfg.suggestedId })}>{label}</button>
            ))}
          </div>
          <span className="field-hint">
            {cfg.role === 'anchor'
              ? 'Broadcasts the beacons receivers measure. One per kit; it sends no data to the laptop.'
              : 'Measures the anchor\'s beacons and streams CSI to this laptop.'}
          </span>
        </div>
        <label className="field">
          <span className="field-label">Node ID</span>
          <input type="number" className="input input-narrow" min={cfg.role === 'anchor' ? 0 : 1} max={255} value={cfg.nodeId}
            onChange={(e) => onChange({ nodeId: e.target.value === '' ? '' : Number(e.target.value) })} />
          <span className="field-hint">
            {doneId != null ? `Installed as node ${doneId}.` : cfg.role === 'anchor' ? 'Anchors use 0.' : `Suggested: ${cfg.suggestedId} (next one not in use).`}
          </span>
          {unknownId && <span className="field-warn"><Icon name="warn" size={14} /> Node {cfg.nodeId} isn't in the kit config, so the backend will ignore its data. Add it under [nodes] or pick {configuredIds.join(', ')}.</span>}
        </label>
        <div className="field field-wide">
          <span className="field-label">Install</span>
          <div className="segmented" role="radiogroup" aria-label="Install type">
            {[['full', 'Firmware + settings'], ['settings', 'Settings only']].map(([v, label]) => (
              <button key={v} type="button" role="radio" aria-checked={cfg.mode === v} className={`seg ${cfg.mode === v ? 'on' : ''}`}
                onClick={() => onChange({ mode: v })}>{label}</button>
            ))}
          </div>
          <span className="field-hint">
            {cfg.mode === 'full' ? 'Erases and installs TRI-FI firmware (about 30 s).' : 'Only rewrites Wi-Fi and node settings (a few seconds). Needs TRI-FI firmware already on the board.'}
          </span>
        </div>
      </fieldset>

      {(running || job.state === 'done' || job.state === 'error') && (
        <div className="board-job" aria-live="polite">
          {running && (
            <>
              <div className="job-line"><span>{job.stage}</span><strong>{Math.round(job.percent ?? 0)}%</strong></div>
              <Meter value={(job.percent ?? 0) / 100} label={`Install progress on ${port.device}`} valueText={`${Math.round(job.percent ?? 0)}%`} />
              <p className="field-hint">Don't unplug the board.</p>
            </>
          )}
          {job.state === 'error' && (
            <div className="inline-alert inline-bad" role="alert"><Icon name="cross" size={16} /><span>{job.error}</span></div>
          )}
          {job.state === 'done' && job.role === 'anchor' && (
            <div className="inline-alert inline-ok"><Icon name="check" size={16} /><span>Anchor installed and restarting. Receivers lock onto it automatically.</span></div>
          )}
          {job.state === 'done' && job.role === 'receiver' && (
            <div className={`inline-alert ${joined ? 'inline-ok' : 'inline-info'}`}>
              <Icon name={joined ? 'check' : 'clock'} size={16} />
              <span>{joined ? `Node ${doneId} is sending data (${link.rate_hz.toFixed(0)} frames/s).` : `Installed. Waiting for node ${doneId} to join Wi-Fi and send data; this usually takes 5–15 s.`}</span>
            </div>
          )}
        </div>
      )}

      <div className="board-actions">
        <button type="button" className="btn btn-primary" onClick={onInstall} disabled={Boolean(blocker) || running}
          title={blocker?.text}>
          <Icon name="bolt" size={15} />
          {running ? 'Installing…' : job.state === 'done' ? 'Install again' : cfg.mode === 'full' ? `Install on ${port.device}` : `Write settings to ${port.device}`}
        </button>
        {blocker && !running && <span className="field-hint">{blocker.text}</span>}
      </div>
    </article>
  );
}
