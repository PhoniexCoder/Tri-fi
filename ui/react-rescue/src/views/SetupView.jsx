import React, { useEffect, useRef, useState } from 'react';
import BoardCard from '../components/BoardCard';
import Icon, { Status } from '../components/Icon';

const PW_OK = (pw) => pw === '' || (pw.length >= 8 && pw.length <= 63);

/** First configured receiver ID not already live, installed, or picked for another board. */
function suggestId(nodeIds, snapshot, flashed, picked) {
  const taken = new Set([
    ...picked,
    ...flashed.filter((f) => f.role === 'receiver').map((f) => f.node_id),
    ...nodeIds.filter((id) => (snapshot?.links?.[id]?.rate_hz ?? 0) >= 1),
  ]);
  const free = nodeIds.find((id) => !taken.has(id));
  return free ?? Math.max(0, ...nodeIds, ...taken) + 1;
}

/**
 * Install TRI-FI on ESP32-S3 boards plugged into this laptop. Boards are
 * detected automatically; the operator types the Wi-Fi name and password,
 * everything else (image size, node ID, laptop address, port) is filled in.
 */
export default function SetupView({ ctx }) {
  const { socket, hello, snapshot, status, setupForm, setSetupForm } = ctx;
  const setup = socket.setup;
  const [boardCfg, setBoardCfg] = useState({});
  const [showPw, setShowPw] = useState(false);
  const requested = useRef(new Set());
  const nodeIds = hello?.node_ids ?? [];
  const ports = setup?.ports ?? [];
  const hosts = setup?.host_ips ?? [];
  const targetIp = setupForm.targetIp || hosts[0]?.ip || '';
  const image = setup?.images?.sets?.['8']?.app ?? setup?.images?.sets?.['4']?.app ?? null;

  // Check each new board once (reads chip, flash size and installed firmware).
  useEffect(() => {
    if (!setup?.available) return;
    const present = new Set(ports.map((p) => p.device));
    for (const d of [...requested.current]) if (!present.has(d)) requested.current.delete(d);
    if (setup.busy) return;
    const next = ports.find((p) => p.probe.state === 'none' && p.job.state !== 'running' && !requested.current.has(p.device));
    if (next && socket.send({ type: 'setup_probe', port: next.device })) requested.current.add(next.device);
  }, [setup]);

  if (status !== 'online') return <Notice icon="wifiOff" title="Backend not connected" text="Start it with python -m trifi run on the laptop the boards are plugged into." />;
  if (!setup) return <Notice icon="replay" title="Setup is off during a replay" text="Installing firmware needs the live backend: python -m trifi run." />;
  if (!setup.available) return <Notice icon="warn" title="Missing tools on this laptop" text={`Install them, then restart the backend: ${setup.reason}`} />;

  const pwOk = PW_OK(setupForm.password);
  const picked = [];
  const cards = ports.map((port) => {
    const own = boardCfg[port.device] ?? {};
    const suggestedId = suggestId(nodeIds, snapshot, setup.flashed ?? [], picked);
    const fw = port.probe.firmware;
    const sameFw = fw && image && fw.project === image.project && fw.version === image.version;
    const cfg = { role: 'receiver', nodeId: suggestedId, mode: sameFw ? 'settings' : 'full', ...own, suggestedId };
    if (cfg.role === 'receiver' && cfg.nodeId !== '') picked.push(cfg.nodeId);
    return { port, cfg };
  });

  const blockerFor = (port, cfg) => {
    if (setup.busy && port.job.state !== 'running') return { text: 'Wait for the other board to finish.', busy: true };
    if (!setupForm.ssid.trim()) return { text: 'Enter the Wi-Fi name first.' };
    if (!pwOk) return { text: 'Wi-Fi password must be 8–63 characters, or empty for an open network.' };
    if (cfg.role === 'receiver' && !targetIp) return { text: 'Choose the laptop address.' };
    if (cfg.nodeId === '' || Number.isNaN(cfg.nodeId)) return { text: 'Enter a node ID.' };
    if (cfg.mode === 'full' && !Object.keys(setup.images?.sets ?? {}).length) return { text: 'No firmware files found on this laptop.' };
    if (port.probe.state === 'error' && cfg.mode === 'full' && port.probe.chip && port.probe.chip !== 'ESP32-S3') return { text: 'This board is not an ESP32-S3.' };
    return null;
  };

  const install = (port, cfg) => {
    socket.clearSetupError();
    // Freeze this board's choices so its node ID doesn't move to the next suggestion afterwards.
    setBoardCfg((m) => ({ ...m, [port.device]: { role: cfg.role, nodeId: cfg.nodeId, mode: cfg.mode } }));
    socket.send({
      type: 'setup_flash', port: port.device, ssid: setupForm.ssid.trim(), password: setupForm.password,
      role: cfg.role, node_id: Number(cfg.nodeId), target_ip: cfg.role === 'receiver' ? targetIp : '',
      target_port: setup.target_port, mode: cfg.mode,
    });
  };

  return (
    <div className="setup-layout">
      <div className="stack">
        <section className="card" aria-labelledby="net-title">
          <header className="card-head">
            <div>
              <h3 id="net-title" className="card-title">Kit Wi-Fi</h3>
              <p className="card-sub">The network every node joins. The laptop's hotspot works anywhere.</p>
            </div>
          </header>
          <div className="form">
            <label className="field">
              <span className="field-label">Wi-Fi name (SSID)</span>
              <input className="input" value={setupForm.ssid} maxLength={32} autoComplete="off" spellCheck={false}
                onChange={(e) => setSetupForm({ ...setupForm, ssid: e.target.value })} placeholder="e.g. TRIFI-KIT" />
            </label>
            <label className="field">
              <span className="field-label">Password</span>
              <span className="input-group">
                <input className="input" type={showPw ? 'text' : 'password'} value={setupForm.password} maxLength={63} autoComplete="new-password"
                  onChange={(e) => setSetupForm({ ...setupForm, password: e.target.value })} aria-invalid={!pwOk} />
                <button type="button" className="btn btn-ghost icon-btn" onClick={() => setShowPw(!showPw)} aria-label={showPw ? 'Hide password' : 'Show password'}>
                  <Icon name={showPw ? 'eyeOff' : 'eye'} size={16} />
                </button>
              </span>
              <span className={pwOk ? 'field-hint' : 'field-warn'}>
                {pwOk ? 'Kept on this screen only until you close it; never saved or logged.' : '8–63 characters, or leave empty for an open network.'}
              </span>
            </label>
            <details className="auto-settings">
              <summary>Filled in automatically: laptop {targetIp || '—'} · port {setup.target_port}</summary>
              <label className="field">
                <span className="field-label">Laptop address the nodes send to</span>
                <select className="input" value={targetIp} onChange={(e) => setSetupForm({ ...setupForm, targetIp: e.target.value })}>
                  {hosts.map((h) => <option key={h.ip} value={h.ip}>{h.ip} ({h.label})</option>)}
                  {!hosts.length && <option value="">No network found</option>}
                </select>
                <span className="field-hint">Pick the address on the Wi-Fi network above. The hotspot is usually 192.168.137.1.</span>
              </label>
              <p className="field-hint">Port {setup.target_port} comes from the backend config.</p>
            </details>
          </div>
        </section>

        <section className="card" aria-labelledby="fw-title">
          <header className="card-head">
            <h3 id="fw-title" className="card-title">Firmware files</h3>
            <Status tone={Object.keys(setup.images?.sets ?? {}).length ? 'ok' : 'bad'}>{Object.keys(setup.images?.sets ?? {}).length ? 'Found' : 'Missing'}</Status>
          </header>
          <dl className="kv-list">
            {['8', '4'].map((mb) => {
              const set = setup.images?.sets?.[mb];
              return (
                <div className="kv" key={mb}>
                  <dt className="nowrap">{mb} MB boards</dt>
                  <dd className="wrap">{set ? `${set.app?.version ?? 'unknown version'}${set.app?.date ? `, built ${set.app.date}` : ''}` : 'Missing'}</dd>
                </div>
              );
            })}
          </dl>
          <p className="note"><Icon name="info" size={14} /><span>From <code className="path">{setup.images?.dir}</code>. To update, put a newer build there (or set <code>[setup] firmware_dir</code>) and restart the backend.</span></p>
          {setup.images?.missing?.length > 0 && <p className="field-warn">Missing: {setup.images.missing.join(', ')}</p>}
        </section>
      </div>

      <section className="stack" aria-labelledby="boards-title">
        <div className="section-head">
          <h2 id="boards-title" className="card-title">Boards on this laptop</h2>
          <span className="card-sub">{ports.length ? `${ports.length} connected · checked automatically` : 'Watching USB…'}</span>
        </div>
        {socket.setupError && (
          <div className="inline-alert inline-bad" role="alert">
            <Icon name="cross" size={16} /><span>{socket.setupError.text}</span>
            <button type="button" className="btn btn-ghost btn-sm" onClick={socket.clearSetupError}>Dismiss</button>
          </div>
        )}
        {cards.map(({ port, cfg }) => (
          <BoardCard key={port.device} port={port} cfg={cfg} image={image} configuredIds={nodeIds} snapshot={snapshot}
            blocker={blockerFor(port, cfg)}
            onChange={(patch) => setBoardCfg((m) => ({ ...m, [port.device]: { ...(m[port.device] ?? {}), ...cfg, ...patch } }))}
            onInstall={() => install(port, cfg)}
            onRetryProbe={() => socket.send({ type: 'setup_probe', port: port.device })} />
        ))}
        {!ports.length && (
          <div className="card empty-boards">
            <span className="empty-icon" aria-hidden="true"><Icon name="usb" size={28} /></span>
            <h3 className="card-title">Plug in an ESP32-S3</h3>
            <p className="card-sub">Use a USB data cable. The board appears here within a couple of seconds.</p>
            <ul className="tips">
              <li>Nothing appears? Many cables only charge. Try another cable.</li>
              <li>Boards with two USB ports: use the one marked USB or COM.</li>
              <li>CH340 boards on Windows may need the CH340 driver once.</li>
            </ul>
          </div>
        )}
      </section>
    </div>
  );
}

function Notice({ icon, title, text }) {
  return (
    <div className="card empty-boards">
      <span className="empty-icon" aria-hidden="true"><Icon name={icon} size={28} /></span>
      <h3 className="card-title">{title}</h3>
      <p className="card-sub">{text}</p>
    </div>
  );
}
