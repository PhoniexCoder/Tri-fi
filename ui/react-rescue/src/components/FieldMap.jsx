import React from 'react';
import { linkState, pct } from '../lib/format';
import { nodeColor, nodeRecords } from '../lib/nodes';
import { useWidth } from './charts/scale';

export const DEFAULT_LAYERS = { heat: true, area: true, survivors: true, rescuer: true, labels: true };
const PAD = 52;

/** Convex hull (monotone chain) of the node positions: the "inside" zone. */
function hull(points) {
  const p = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const pt of p) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], pt) <= 0) lower.pop(); lower.push(pt); }
  const upper = [];
  for (const pt of [...p].reverse()) { while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], pt) <= 0) upper.pop(); upper.push(pt); }
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

/**
 * Search-area map in the backend's unit coordinates (0..1 on both axes).
 * Positions are estimates from signal strength, so people are drawn as
 * areas, not pins. Every mark has a text label or a tooltip; colour is never
 * the only cue.
 */
export default function FieldMap({ snapshot, hello, heatmap, layers = DEFAULT_LAYERS, maxHeight = 520 }) {
  const [ref, width] = useWidth(520);
  const nodeIds = hello?.node_ids ?? [];
  const positions = hello?.positions ?? {};
  const height = Math.min(maxHeight, Math.max(260, Math.round(width * 0.78)));
  const plotW = width - 2 * PAD;
  const plotH = height - 2 * PAD;
  const sx = (v) => PAD + v * plotW;
  const sy = (v) => PAD + v * plotH;
  const records = nodeRecords(snapshot, nodeIds);
  const survivors = snapshot?.ai_detected ? snapshot.survivors_list ?? [] : [];
  const r = snapshot?.rescuer;
  const zone = hull(nodeIds.map((id) => positions[id]).filter(Boolean));
  const cell = heatmap ? Math.sqrt(heatmap.points.length) : 0;
  // Backend values saturate near 1 across most of the area, so draw them
  // relative to this frame's range: the map shows where to look first.
  const hv = heatmap ? heatmap.points.map((p) => p.v) : [];
  const hMin = hv.length ? Math.min(...hv) : 0;
  const hSpan = hv.length ? Math.max(...hv) - hMin : 0;
  const rel = (v) => (hSpan > 1e-6 ? (v - hMin) / hSpan : 0);
  const cw = cell > 1 ? plotW / (cell - 1) : 0;
  const ch = cell > 1 ? plotH / (cell - 1) : 0;
  const personR = Math.max(18, Math.min(plotW, plotH) * 0.09);
  const small = width < 420;
  const badgeAtConfirm = r?.confirmed && r.conf_x != null && Math.hypot(r.x - r.conf_x, r.y - r.conf_y) < 0.08;

  const anchor = (px) => (px < 0.15 ? 'start' : px > 0.85 ? 'end' : 'middle');
  const labelX = (px) => sx(px) + (px < 0.15 ? -14 : px > 0.85 ? 14 : 0);

  return (
    <div className="map" ref={ref}>
      <svg width={width} height={height} role="img"
        aria-label={`Search area map with ${nodeIds.length} nodes${survivors.length ? ` and ${survivors.length} possible ${survivors.length === 1 ? 'person' : 'people'}` : ''}`}>
        <rect className="map-bg" x={PAD - 20} y={PAD - 20} width={plotW + 40} height={plotH + 40} rx="10" />
        {[0, 0.25, 0.5, 0.75, 1].map((g) => (
          <g key={g}>
            <line className="grid" x1={sx(g)} x2={sx(g)} y1={PAD} y2={PAD + plotH} />
            <line className="grid" x1={PAD} x2={PAD + plotW} y1={sy(g)} y2={sy(g)} />
          </g>
        ))}

        {layers.heat && heatmap && heatmap.points.map((p) => {
          const k = rel(p.v);
          return k > 0.08 ? (
            <rect key={`${p.x}-${p.y}`} className="heat" x={sx(p.x) - cw / 2} y={sy(p.y) - ch / 2} width={cw + 0.5} height={ch + 0.5}
              opacity={0.05 + 0.6 * k ** 1.6} />
          ) : null;
        })}

        {layers.area && zone.length >= 3 && (
          <g>
            <polygon className="zone" points={zone.map(([x, y]) => `${sx(x)},${sy(y)}`).join(' ')} />
          </g>
        )}

        {records.map(({ id, node, link }) => {
          const pos = positions[id];
          if (!pos) return null;
          const [px, py] = pos;
          const health = linkState(link);
          const detecting = node?.csi_detected;
          const below = py < 0.85;
          return (
            <g key={id} className={`map-node ${health.key === 'down' ? 'is-down' : ''}`} tabIndex={0}
              aria-label={`Node ${id}: ${health.label}${detecting ? ', detecting' : ''}, model ${pct(node?.csi_ml_prob)}`}>
              <title>{`Node ${id} · ${health.label}\nModel ${pct(node?.csi_ml_prob)} · ${link?.rate_hz ?? 0} frames/s · ${link?.rssi ?? '—'} dBm`}</title>
              {detecting && <circle className="detect-ring" cx={sx(px)} cy={sy(py)} r="22" />}
              <circle cx={sx(px)} cy={sy(py)} r="15" fill={nodeColor(nodeIds, id)} className="ring node-dot" />
              <text className={`node-num ${nodeIds.indexOf(id) === 3 ? 'on-light' : ''}`} x={sx(px)} y={sy(py)} dy="0.35em" textAnchor="middle">{id}</text>
              {layers.labels && (
                <text className="map-label" x={labelX(px)} y={sy(py) + (below ? 38 : -30)} textAnchor={anchor(px)}>
                  <tspan className={`tone-text-${health.tone}`}>{health.key === 'good' ? '●' : health.key === 'weak' ? '▲' : '✕'}</tspan>
                  {` Node ${id}`}
                  {!small && <tspan className="map-label-sub">{` · ${detecting ? 'detecting' : health.label.toLowerCase()}`}</tspan>}
                </text>
              )}
            </g>
          );
        })}

        {layers.rescuer && r?.confirmed && r.conf_x != null && (
          <g className="map-confirmed" tabIndex={0} aria-label="Rescue confirmed at this spot">
            <title>Rescue confirmed here by the rescuer badge</title>
            <circle cx={sx(r.conf_x)} cy={sy(r.conf_y)} r="11" className="ring" />
            <path d={`M${sx(r.conf_x) - 5},${sy(r.conf_y)} l3.5,3.5 l6.5,-7`} className="check" />
            <text className="map-label" x={sx(r.conf_x)} y={sy(r.conf_y) + 28} textAnchor="middle">
              {r.active && badgeAtConfirm ? 'Rescuer · rescue confirmed' : 'Rescue confirmed'}
            </text>
          </g>
        )}
        {layers.rescuer && r?.active && !badgeAtConfirm && (
          <g className="map-rescuer" tabIndex={0} aria-label="Rescuer badge position (approximate)">
            <title>Rescuer badge, approximate position from node signal strength</title>
            <rect x={sx(r.x) - 8} y={sy(r.y) - 8} width="16" height="16" transform={`rotate(45 ${sx(r.x)} ${sy(r.y)})`} className="ring" />
            <text className="map-label" x={sx(r.x)} y={sy(r.y) - 18} textAnchor="middle">Rescuer</text>
          </g>
        )}

        {layers.survivors && survivors.map((s, i) => {
          const cx = sx(s.x);
          const cy = sy(s.y);
          const above = s.y > 0.2;
          return (
            <g key={i} className="map-person" tabIndex={0} aria-label={`Possible person ${i + 1}, ${pct(s.intensity)} confidence, approximate area`}>
              <title>{`Possible person ${i + 1}\nConfidence ${pct(s.intensity)}\nApproximate area: position is estimated from signal strength`}</title>
              <circle cx={cx} cy={cy} r={personR} className="person-area" />
              <circle cx={cx} cy={cy} r="5" className="person-dot ring" />
              <g transform={`translate(${Math.min(Math.max(cx, 80), width - 80)} ${above ? cy - personR - 12 : cy + personR + 22})`}>
                <rect className="person-pill" x={small ? -54 : -78} y="-13" width={small ? 108 : 156} height="22" rx="11" />
                <text className="person-text" textAnchor="middle" dy="0.1em">
                  {small ? `Person? ${pct(s.intensity)}` : `Possible person · ${pct(s.intensity)}`}
                </text>
              </g>
            </g>
          );
        })}

        {!nodeIds.length && (
          <text className="empty" x={width / 2} y={height / 2} textAnchor="middle">Waiting for the backend to send node positions</text>
        )}
      </svg>
      <MapLegend heat={layers.heat && Boolean(heatmap)} />
    </div>
  );
}

function MapLegend({ heat }) {
  return (
    <ul className="map-legend" aria-label="Map legend">
      <li><span className="lg-node" /> Node (number inside)</li>
      <li><span className="lg-detect" /> Node detecting</li>
      <li><span className="lg-person" /> Possible person (approx. area)</li>
      <li><span className="lg-zone" /> Inside node area</li>
      <li><span className="lg-rescuer" /> Rescuer badge</li>
      {heat && (
        <li className="lg-heat-item">
          <span className="lg-heat" /> Likelihood (relative), low → high
        </li>
      )}
    </ul>
  );
}
