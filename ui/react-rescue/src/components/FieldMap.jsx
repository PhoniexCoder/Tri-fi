import React from 'react';
import { linkState } from '../lib/format';
import { MAP, posToSvg } from '../lib/geometry';

/**
 * Nodes are drawn at the positions the backend uses and coloured by link
 * health. A detection is drawn as an approximate area: positions are a
 * weighted estimate between nodes, not a measured fix. The rescuer badge is
 * not drawn because its position estimate uses the wrong signal (Phase 4).
 */
export default function FieldMap({ snapshot, hello }) {
  const positions = hello?.positions ?? {};
  const nodeIds = (hello?.node_ids ?? []).filter((id) => positions[id]);
  const pts = nodeIds.map((id) => ({ id, ...posToSvg(...positions[id]) }));
  const links = snapshot?.links ?? {};
  const detectingIds = new Set((snapshot?.nodes ?? []).filter((n) => n.csi_detected).map((n) => n.node_id));
  const detected = snapshot?.ai_detected ?? false;
  const survivors = detected ? snapshot?.survivors_list ?? [] : [];
  const cx = pts.reduce((s, p) => s + p.x, 0) / (pts.length || 1);
  const cy = pts.reduce((s, p) => s + p.y, 0) / (pts.length || 1);

  return (
    <section className="card map-card">
      <div className="card-head">
        <h2>Search area</h2>
        <span className="card-note">Positions are approximate</span>
      </div>
      <svg className="map" viewBox={`0 0 ${MAP.w} ${MAP.h}`} role="img"
           aria-label={detected ? 'Map: possible survivor inside the node area' : 'Map: no detection'}>
        {pts.length >= 3 && (
          <polygon className={`map-area ${detected ? 'is-alert' : ''}`}
                   points={pts.map((p) => `${p.x},${p.y}`).join(' ')} />
        )}

        {pts.map((p) => {
          const state = linkState(links[p.id]);
          return (
            <g key={p.id} className={`map-node tone-${state.tone}`}>
              {detectingIds.has(p.id) && <circle className="map-ring" cx={p.x} cy={p.y} r="34" />}
              <circle className="map-node-dot" cx={p.x} cy={p.y} r="20" />
              <text className="map-node-num" x={p.x} y={p.y + 6} textAnchor="middle">{p.id}</text>
              <text className="map-node-label" x={p.x} y={p.y > cy ? p.y + 58 : p.y - 46} textAnchor="middle">
                Node {p.id}
                <tspan className="map-node-status"> · {state.label}</tspan>
              </text>
            </g>
          );
        })}

        {survivors.map((s, i) => {
          const { x, y } = posToSvg(s.x, s.y);
          return (
            <g key={i} className="map-survivor">
              <circle className="map-survivor-area" cx={x} cy={y} r="46" />
              <circle className="map-survivor-pulse" cx={x} cy={y} r="12" />
              <circle className="map-survivor-dot" cx={x} cy={y} r="9" />
              <text className="map-survivor-label" x={x} y={y > cy ? y - 58 : y + 68} textAnchor="middle">
                Possible survivor (approx. area)
              </text>
            </g>
          );
        })}

        {!snapshot && (
          <text className="map-empty" x={MAP.w / 2} y={MAP.h / 2} textAnchor="middle">Waiting for data…</text>
        )}
      </svg>
      <div className="map-legend">
        <span><span className="swatch swatch-ok" /> node, good link</span>
        <span><span className="swatch swatch-warn" /> weak link</span>
        <span><span className="swatch swatch-bad" /> no data</span>
        <span><span className="swatch swatch-ring" /> node detecting</span>
        <span><span className="swatch swatch-alert" /> possible survivor</span>
      </div>
    </section>
  );
}
