import React from 'react';
import { nodeColor, posToSvg } from '../lib/geometry';

/**
 * Node markers are drawn at the positions the backend uses (from the hello
 * message), so detections and nodes share one coordinate system.
 * Positions are a weighted average of node locations, not a measured fix.
 * The rescuer badge is not drawn: its position estimate uses the wrong signal
 * (known issue, Phase 4).
 */
export default function FieldMap({ snapshot, hello }) {
  const positions = hello?.positions ?? {};
  const nodeIds = (hello?.node_ids ?? []).filter((id) => positions[id]);
  const points = nodeIds.map((id) => ({ id, ...posToSvg(...positions[id]) }));
  const activeIds = new Set((snapshot?.nodes ?? []).filter((n) => n.active).map((n) => n.node_id));
  const detected = snapshot?.ai_detected ?? false;
  const survivors = detected ? snapshot?.survivors_list ?? [] : [];

  return (
    <div id="triangle-wrap">
      <svg id="triangle-svg" viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <filter id="glow">
            <feGaussianBlur stdDeviation="4" result="blur" />
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

        {[100, 200, 300, 400].map((v) => (
          <g key={v}>
            <line x1="0" y1={v} x2="500" y2={v} className="grid-line" />
            <line x1={v} y1="0" x2={v} y2="500" className="grid-line" />
          </g>
        ))}

        {points.length >= 3 && (
          <polygon
            className={`tri-fill ${detected ? 'detected' : ''}`}
            points={points.map((p) => `${p.x},${p.y}`).join(' ')}
          />
        )}

        {survivors.map((s, i) => {
          const { x, y } = posToSvg(s.x, s.y);
          return (
            <g key={i}>
              <circle cx={x} cy={y} r="70" fill="url(#heat-grad)" opacity={Math.min(s.intensity ?? 0.6, 1)} />
              <circle
                className="survivor-pulse"
                cx={x}
                cy={y}
                r="12"
                style={{ animation: `pulse-expand 1.5s infinite ${i * 0.3}s` }}
              />
              <circle className="survivor-core" cx={x} cy={y} r="12" />
            </g>
          );
        })}

        {points.map((p) => (
          <g key={p.id} opacity={activeIds.has(p.id) ? 1 : 0.3}>
            <circle className="range-ring" cx={p.x} cy={p.y} r="90" opacity="0.6" />
            <g filter="url(#glow)">
              <circle cx={p.x} cy={p.y} r="14" fill="#0d1421" stroke={nodeColor(p.id)} strokeWidth="2.5" />
              <circle cx={p.x} cy={p.y} r="6" fill={nodeColor(p.id)} />
            </g>
            <text
              x={p.x}
              y={p.y > 250 ? p.y + 32 : p.y - 22}
              textAnchor="middle"
              fontFamily="Space Mono,monospace"
              fontSize="11"
              fill={nodeColor(p.id)}
            >
              N{p.id}
            </text>
          </g>
        ))}
      </svg>
      <div className="map-footer">
        <div className="map-label">
          CSI COVERAGE MAP — position is a weighted estimate between nodes, not a measured fix
        </div>
      </div>
    </div>
  );
}
