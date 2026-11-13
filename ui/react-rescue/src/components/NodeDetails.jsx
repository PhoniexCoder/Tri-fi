import React from 'react';
import { nodeColor, pct } from '../lib/geometry';

export default function NodeDetails({ snapshot, nodeIds }) {
  const nodes = snapshot?.nodes ?? [];
  return (
    <>
      <div className="panel-title">Node Details</div>
      {nodeIds.map((id) => {
        const node = nodes.find((n) => n.node_id === id) ?? {};
        const active = node.active ?? false;
        return (
          <div key={id} className="node-card" style={active ? { borderColor: nodeColor(id) } : undefined}>
            <div className="node-card-header">
              <span className="node-card-title" style={{ color: nodeColor(id) }}>
                ▶ NODE {id}
              </span>
              <span className={`node-status-pill ${active ? 'online' : ''}`}>
                {active ? 'ONLINE' : 'OFFLINE'}
              </span>
            </div>
            <div className="node-metric">
              <span className="lbl">RSSI</span>
              <span className="val">{(node.rssi ?? -100).toFixed(0)} dBm</span>
            </div>
            <div className="node-metric">
              <span className="lbl">CSI model</span>
              <span className="val">
                {pct(node.csi_ml_prob)} {node.csi_detected ? '• detecting' : ''}
              </span>
            </div>
            <div className="node-metric">
              <span className="lbl">Breathing (FFT)</span>
              <span className="val" style={{ opacity: node.breathing_detected ? 1 : 0.6 }}>
                {node.breathing > 0 ? `${node.breathing.toFixed(1)} rpm` : '-- rpm'}
              </span>
            </div>
            <div className="node-metric">
              <span className="lbl">Motion (on-chip)</span>
              <span className="val">{(node.motion ?? 0).toFixed(3)}</span>
            </div>
          </div>
        );
      })}
    </>
  );
}
