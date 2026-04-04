import React, { useEffect, useState, useMemo } from 'react';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8002';

function rssiToSigStrength(rssi) {
  return Math.max(0, Math.min(100, ((rssi + 100) / 60) * 100));
}

function posToSvg(px, py) {
  const x = 80 + px * (420 - 80);
  const y = 80 + py * (440 - 80);
  return { x, y };
}

function App() {
  const [connected, setConnected] = useState(false);
  const [payload, setPayload] = useState(null);

  useEffect(() => {
    let ws;
    function connect() {
      ws = new WebSocket(WS_URL);
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        setTimeout(connect, 2000);
      };
      ws.onerror = () => {};
      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          if (data.type === 'rescue_ai') setPayload(data);
        } catch {
          // ignore
        }
      };
    }
    connect();
    return () => {
      if (ws) ws.close();
    };
  }, []);

  const ui = useMemo(() => {
    if (!payload) return null;
    const detected = payload.ai_detected;
    const prob = payload.ai_prob ?? 0;
    const activeNodes = payload.active_nodes ?? 0;
    const nodes = payload.nodes ?? [];
    const px = payload.pos_x ?? 0.5;
    const py = payload.pos_y ?? 0.5;
    const mode = payload.ai_mode || '—';
    const survivorsCount = payload.survivors ?? 0;
    const csiProb = payload.csi_ml_prob ?? 0;
    const triProb = payload.tri_prob ?? 0;
    const survivorsList = payload.survivors_list ?? [];

    let avgRssi = null;
    let maxMotion = null;
    let avgBr = null;
    const active = nodes.filter((n) => n.active);
    if (active.length > 0) {
      avgRssi = active.reduce((s, n) => s + n.rssi, 0) / active.length;
      maxMotion = Math.max(...active.map((n) => n.motion));

      // Only treat breathing as valid when the backend breathing
      // detector has actually fired for that node. This prevents
      // a constant-looking RPM value when there is no clear
      // breathing signal.
      const breathingNodes = active.filter((n) => n.breathing_detected);
      if (breathingNodes.length > 0) {
        avgBr =
          breathingNodes.reduce((s, n) => s + n.breathing, 0) /
          breathingNodes.length;
      }
    }

    return {
      detected,
      prob,
      activeNodes,
      nodes,
      px,
      py,
      mode,
      survivorsCount,
      survivorsList,
      csiProb,
      triProb,
      avgRssi,
      maxMotion,
      avgBr,
    };
  }, [payload]);

  const heat = useMemo(() => {
    if (!ui) return { x: 250, y: 260, opacity: 0 };

    const survivors = ui.survivorsList ?? [];
    if (ui.detected && survivors.length > 0) {
      let sumX = 0;
      let sumY = 0;
      let maxIntensity = 0;

      survivors.forEach((s) => {
        const { x, y } = posToSvg(s.x, s.y);
        sumX += x;
        sumY += y;
        if (typeof s.intensity === 'number') {
          maxIntensity = Math.max(maxIntensity, s.intensity);
        }
      });

      const avgX = sumX / survivors.length;
      const avgY = sumY / survivors.length;
      const opacity = maxIntensity > 0 ? Math.min(maxIntensity * 0.8, 1) : 0.6;
      return { x: avgX, y: avgY, opacity };
    }

    const { x, y } = posToSvg(ui.px ?? 0.5, ui.py ?? 0.5);
    return { x, y, opacity: ui.detected ? 0.6 : 0 };
  }, [ui]);

  const node1Active = ui?.nodes?.some((n) => n.node_id === 1 && n.active) ?? false;
  const node2Active = ui?.nodes?.some((n) => n.node_id === 2 && n.active) ?? false;
  const node3Active = ui?.nodes?.some((n) => n.node_id === 3 && n.active) ?? false;

  return (
    <div className="app">
      <header id="header">
        <div className="logo">
          <div className="logo-text">
            <h1>TRIFI</h1>
            <span>Real-time WiFi Survivor Search • Field Ops Console</span>
          </div>
        </div>
        <div className="header-right">
          <div className="incident-meta">
            <div className="incident-label">OP MODE</div>
            <div className="incident-value">LIVE DEPLOYMENT</div>
          </div>
          <div className="header-divider" />
          <div className="node-count">
            NODES:{' '}
            <span
              id="val-nodes"
              style={{
                color: ui
                  ? ui.activeNodes === 3
                    ? 'var(--green)'
                    : ui.activeNodes > 0
                    ? 'var(--orange)'
                    : 'var(--red)'
                  : 'var(--muted)',
              }}
            >
              {ui?.activeNodes ?? 0}
            </span>{' '}
            / 3
          </div>
          <span id="mode-badge">{(ui?.mode ?? 'LOADING...').toUpperCase()}</span>
          <div id="ws-badge" className={`ws-badge ${connected ? 'online' : ''}`}>
            {connected ? 'LINK STABLE' : 'LINK LOST'}
          </div>
        </div>
      </header>

      <div id="main">
        <div id="panel-left">
          <div className="panel-title">Situation Overview</div>

          <div
            id="presence-card"
            className={ui?.detected ? 'detected' : ''}
          >
            <div
              id="presence-label"
              className={ui?.detected ? 'detected' : ''}
            >
              {ui?.detected
                ? ui.survivorsCount > 1
                  ? `⚠ MULTIPLE SURVIVORS (${ui.survivorsCount})`
                  : '⚠ SURVIVOR DETECTED'
                : 'NO SURVIVOR'}
            </div>
            <div id="confidence-bar-wrap">
              <div className="bar-label">
                <span>Confidence</span>
                <span id="val-prob-pct">{((ui?.prob ?? 0) * 100).toFixed(1)}%</span>
              </div>
              <div className="bar-track">
                <div
                  className="bar-fill"
                  id="conf-bar"
                  style={{ width: `${(ui?.prob ?? 0) * 100}%` }}
                />
              </div>
            </div>
            <div id="summary-grid">
              <div className="summary-item">
                <div className="summary-label">Mode</div>
                <div className="summary-value" id="summary-mode">
                  {(ui?.mode ?? '—').toUpperCase()}
                </div>
              </div>
              <div className="summary-item">
                <div className="summary-label">Survivors</div>
                <div className="summary-value" id="summary-survivors">
                  {ui?.detected ? ui.survivorsCount : 0}
                </div>
              </div>
              <div className="summary-item">
                <div className="summary-label">CSI Model</div>
                <div className="summary-value" id="summary-csi-prob">
                  {((ui?.csiProb ?? 0) * 100).toFixed(1)}%
                </div>
              </div>
              <div className="summary-item">
                <div className="summary-label">Triangle</div>
                <div className="summary-value" id="summary-tri-prob">
                  {((ui?.triProb ?? 0) * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>

          <div className="panel-title">Team-wide Signal Picture</div>
          <div>
            <div className="metric-row">
              <span className="metric-label">Avg RSSI</span>
              <span className="metric-val" id="val-rssi">
                {ui?.avgRssi != null ? `${ui.avgRssi.toFixed(1)} dBm` : '-- dBm'}
              </span>
            </div>
            <div className="metric-row">
              <span className="metric-label">Max Motion</span>
              <span className="metric-val" id="val-motion">
                {ui?.maxMotion != null ? ui.maxMotion.toFixed(3) : '--'}
              </span>
            </div>
            <div className="metric-row">
              <span className="metric-label">Breathing</span>
              <span className="metric-val" id="val-br">
                {ui?.avgBr != null ? `${ui.avgBr.toFixed(1)} rpm` : '-- rpm'}
              </span>
            </div>
          </div>

          <div className="panel-title">Node Health</div>
          <div>
            {[1, 2, 3].map((n) => {
              const node = ui?.nodes?.find((x) => x.node_id === n);
              const active = node?.active ?? false;
              const rssi = node?.rssi ?? -100;
              const motion = node?.motion ?? 0;
              const breathing = node?.breathing ?? 0;
              const pct = rssiToSigStrength(rssi);
              return (
                <div className="node-bar" key={n} id={`node-bar-${n}`}>
                  <div
                    className={`node-dot n${n} ${!active ? 'offline' : ''}`}
                  />
                  <div className="node-info">
                    <div className="node-name">
                      {n === 1
                        ? 'NODE 1 (BOTTOM)'
                        : n === 2
                        ? 'NODE 2 (TOP-LEFT)'
                        : 'NODE 3 (TOP-RIGHT)'}
                    </div>
                    <div className="node-signal-bar">
                      <div
                        className="node-signal-fill"
                        id={`sig-${n}`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                  <span
                    style={{
                      fontFamily: 'monospace',
                      fontSize: 11,
                      color: 'var(--muted)',
                    }}
                    id={`rssi-${n}`}
                  >
                    {rssi.toFixed(0)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        <div id="center">
          <div
            id="alert-banner"
            style={{ display: ui?.detected ? 'block' : 'none' }}
          >
            {ui?.detected
              ? ui.survivorsCount > 1
                ? `⚠ MULTIPLE SURVIVORS (${ui.survivorsCount})`
                : '⚠ SURVIVOR DETECTED'
              : ''}
          </div>

          <div id="triangle-wrap">
            <svg
              id="triangle-svg"
              viewBox="0 0 500 500"
              xmlns="http://www.w3.org/2000/svg"
            >
              <defs>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="4" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
                <filter id="glow-red">
                  <feGaussianBlur stdDeviation="6" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
                <radialGradient id="heat-grad" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity="0.5" />
                  <stop offset="100%" stopColor="#ef4444" stopOpacity="0" />
                </radialGradient>
              </defs>

              {/* background grid */}
              {[100, 200, 300, 400].map((y) => (
                <line
                  key={`h-${y}`}
                  x1="0"
                  y1={y}
                  x2="500"
                  y2={y}
                  className="grid-line"
                />
              ))}
              {[100, 200, 300, 400].map((x) => (
                <line
                  key={`v-${x}`}
                  x1={x}
                  y1="0"
                  x2={x}
                  y2="500"
                  className="grid-line"
                />
              ))}

              <polygon
                id="tri-poly"
                className={`tri-fill ${ui?.detected ? 'detected' : ''}`}
                points="250,440 80,80 420,80"
              />

              {/* range rings */}
              <circle cx="250" cy="440" r="90" className="range-ring" opacity="0.85" />
              <circle cx="250" cy="440" r="160" className="range-ring" opacity="0.55" />
              <circle cx="80" cy="80" r="90" className="range-ring" opacity="0.85" />
              <circle cx="80" cy="80" r="160" className="range-ring" opacity="0.55" />
              <circle cx="420" cy="80" r="90" className="range-ring" opacity="0.85" />
              <circle cx="420" cy="80" r="160" className="range-ring" opacity="0.55" />

              {/* continuous wave animation around nodes */}
              <g className="range-waves">
                <circle
                  className="wave-ring wave-n1"
                  cx="250"
                  cy="440"
                  r="40"
                />
                <circle
                  className="wave-ring wave-n2"
                  cx="80"
                  cy="80"
                  r="40"
                />
                <circle
                  className="wave-ring wave-n3"
                  cx="420"
                  cy="80"
                  r="40"
                />
              </g>

              <circle
                id="heat-glow"
                cx={heat.x}
                cy={heat.y}
                r="70"
                fill="url(#heat-grad)"
                opacity={heat.opacity}
              />

              {ui?.detected && ui.survivorsList && ui.survivorsList.length > 0 && (
                <g id="survivors-layer">
                  {ui.survivorsList.map((s, idx) => {
                    const { x, y } = posToSvg(s.x, s.y);
                    const delay = `${idx * 0.3}s`;
                    return (
                      <g key={idx}>
                        <circle
                          className="survivor-pulse"
                          cx={x}
                          cy={y}
                          r="12"
                          style={{ animation: `pulse-expand 1.5s infinite ${delay}` }}
                        />
                        <circle
                          className="survivor-core"
                          cx={x}
                          cy={y}
                          r="12"
                        />
                      </g>
                    );
                  })}
                </g>
              )}

              {/* nodes */}
              <g id="node-1-g" filter="url(#glow)" opacity={node1Active ? 1 : 0.3}>
                <circle
                  cx="250"
                  cy="440"
                  r="14"
                  fill="#0d1421"
                  stroke="#00d4ff"
                  strokeWidth="2.5"
                />
                <circle cx="250" cy="440" r="6" fill="#00d4ff" />
              </g>
              <text
                x="250"
                y="472"
                textAnchor="middle"
                fontFamily="Space Mono,monospace"
                fontSize="11"
                fill="#00d4ff"
              >
                N1
              </text>

              <g id="node-2-g" filter="url(#glow)" opacity={node2Active ? 1 : 0.3}>
                <circle
                  cx="80"
                  cy="80"
                  r="14"
                  fill="#0d1421"
                  stroke="#7c3aed"
                  strokeWidth="2.5"
                />
                <circle cx="80" cy="80" r="6" fill="#7c3aed" />
              </g>
              <text
                x="80"
                y="58"
                textAnchor="middle"
                fontFamily="Space Mono,monospace"
                fontSize="11"
                fill="#7c3aed"
              >
                N2
              </text>

              <g id="node-3-g" filter="url(#glow)" opacity={node3Active ? 1 : 0.3}>
                <circle
                  cx="420"
                  cy="80"
                  r="14"
                  fill="#0d1421"
                  stroke="#10b981"
                  strokeWidth="2.5"
                />
                <circle cx="420" cy="80" r="6" fill="#10b981" />
              </g>
              <text
                x="420"
                y="58"
                textAnchor="middle"
                fontFamily="Space Mono,monospace"
                fontSize="11"
                fill="#10b981"
              >
                N3
              </text>
            </svg>
          </div>

          <div className="map-footer">
            <div className="map-label">RF REFLECTION FIELD — TRIANGULAR COVERAGE MAP</div>
            <div className="legend">
              <span className="legend-dot legend-n1" /> N1 • BOTTOM (COMMAND)
              <span className="legend-separator">|</span>
              <span className="legend-dot legend-n2" /> N2 • TOP-LEFT (LEFT FLANK)
              <span className="legend-separator">|</span>
              <span className="legend-dot legend-n3" /> N3 • TOP-RIGHT (RIGHT FLANK)
            </div>
          </div>
        </div>

        <div id="panel-right">
          <div className="panel-title">Node Details</div>
          {[1, 2, 3].map((n) => {
            const node = ui?.nodes?.find((x) => x.node_id === n);
            const active = node?.active ?? false;
            const rssi = node?.rssi ?? -100;
            const motion = node?.motion ?? 0;
            const breathing = node?.breathing ?? 0;
            const breathingDetected = node?.breathing_detected ?? false;
            return (
              <div
                key={n}
                className={`node-card ${active ? `active-n${n}` : ''}`}
                id={`card-n${n}`}
              >
                <div className="node-card-header">
                  <span
                    className="node-card-title"
                    style={{
                      color:
                        n === 1
                          ? 'var(--node1)'
                          : n === 2
                          ? 'var(--node2)'
                          : 'var(--node3)',
                    }}
                  >
                    {`▶ NODE ${n} — ${
                      n === 1 ? 'BOTTOM' : n === 2 ? 'TOP-LEFT' : 'TOP-RIGHT'
                    }`}
                  </span>
                  <span
                    className={`node-status-pill ${active ? 'online' : ''}`}
                    id={`pill-${n}`}
                  >
                    {active ? 'ONLINE' : 'OFFLINE'}
                  </span>
                </div>
                <div className="node-metric">
                  <span className="lbl">RSSI</span>
                  <span className="val" id={`c-rssi-${n}`}>
                    {rssi.toFixed(1)} dBm
                  </span>
                </div>
                <div className="node-metric">
                  <span className="lbl">Motion</span>
                  <span className="val" id={`c-mot-${n}`}>
                    {motion.toFixed(3)}
                  </span>
                </div>
                <div className="node-metric">
                  <span className="lbl">Breathing</span>
                  <span className="val" id={`c-br-${n}`}>
                    {breathingDetected ? `${breathing.toFixed(1)} rpm` : '-- rpm'}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default App;
