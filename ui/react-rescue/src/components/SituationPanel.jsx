import React from 'react';
import { pct } from '../lib/geometry';

/**
 * Detection summary. The bar shows the raw combined model output with the
 * detection threshold marked, rather than a rescaled value.
 */
export default function SituationPanel({ snapshot }) {
  const detected = snapshot?.ai_detected ?? false;
  const survivors = snapshot?.survivors ?? 0;
  const confidence = snapshot?.confidence_raw ?? 0;
  const threshold = snapshot?.ui_threshold ?? 0.3;
  const modelLoaded = snapshot?.model?.loaded ?? false;
  const label = detected
    ? survivors > 1
      ? `⚠ POSSIBLE SURVIVORS (${survivors})`
      : '⚠ POSSIBLE SURVIVOR'
    : 'NO DETECTION';

  return (
    <>
      <div className="panel-title">Situation Overview</div>
      <div id="presence-card" className={detected ? 'detected' : ''}>
        <div id="presence-label" className={detected ? 'detected' : ''}>
          {label}
        </div>
        <div id="confidence-bar-wrap">
          <div className="bar-label">
            <span>Model output</span>
            <span id="val-prob-pct">{pct(confidence)}</span>
          </div>
          <div className="bar-track">
            <div className="bar-fill" id="conf-bar" style={{ width: pct(confidence) }} />
            <div className="threshold-marker" style={{ left: pct(threshold) }} title="Detection threshold" />
          </div>
          <div className="bar-caption">
            Threshold {pct(threshold, 0)} · {snapshot?.consensus_status ?? '—'} ·{' '}
            {snapshot?.n_detecting ?? 0} node(s) agree
          </div>
        </div>
        <div id="summary-grid">
          <div className="summary-item">
            <div className="summary-label">Survivors</div>
            <div className="summary-value">{detected ? survivors : 0}</div>
          </div>
          <div className="summary-item">
            <div className="summary-label">CSI model</div>
            <div className="summary-value">{modelLoaded ? pct(snapshot?.csi_ml_prob) : 'OFF'}</div>
          </div>
          <div className="summary-item">
            <div className="summary-label">Legacy RSSI</div>
            <div className="summary-value">{pct(snapshot?.tri_prob)}</div>
          </div>
          <div className="summary-item">
            <div className="summary-label">Active nodes</div>
            <div className="summary-value">{snapshot?.active_nodes ?? 0}</div>
          </div>
        </div>
        <div className="disclaimer">A detection is a lead to confirm, never proof an area is clear.</div>
      </div>
    </>
  );
}
