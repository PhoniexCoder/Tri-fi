import React from 'react';
import { consensusText, duration, pct } from '../lib/format';
import Icon from './Icon';

/**
 * The one thing an operator must see at a glance: is there a possible
 * survivor, how sure is the system, and what to do next.
 */
export default function StatusHero({ snapshot, ready, detectedSince }) {
  const detected = snapshot?.ai_detected ?? false;
  const confidence = snapshot?.confidence_raw ?? 0;
  const threshold = snapshot?.ui_threshold ?? 0.3;
  const agree = snapshot?.n_detecting ?? 0;
  const total = snapshot?.nodes?.length ?? 0;
  const survivors = snapshot?.survivors ?? 0;

  let tone = 'idle';
  let title = 'Waiting for data';
  let advice = 'Start the backend and power on the nodes.';
  if (snapshot) {
    if (detected) {
      tone = 'alert';
      title = survivors > 1 ? `Possible survivors detected (${survivors})` : 'Possible survivor detected';
      advice = 'Confirm with a call-out, acoustic sensor or camera before committing a team.';
    } else if (!ready) {
      tone = 'warn';
      title = 'Not ready to search';
      advice = 'Check the readiness list below. Keep the area clear while nodes calibrate.';
    } else {
      // Deliberately not green: "nothing detected" must never read as "all clear".
      tone = 'search';
      title = 'Searching — no survivor detected';
      advice = 'No detection does not mean the area is clear. Keep other search methods going.';
    }
  }
  const icon = { alert: 'warn', warn: 'warn', search: 'pulse', idle: 'info' }[tone];

  return (
    <section className={`hero hero-${tone}`} aria-live="polite">
      <div className="hero-state">
        <span className="hero-icon">
          <Icon name={icon} size={34} />
        </span>
        <div>
          <h1 className="hero-title">{title}</h1>
          <p className="hero-advice">{advice}</p>
        </div>
      </div>

      <div className="hero-stats">
        <div className="stat stat-confidence">
          <div className="stat-label">Model confidence</div>
          <div className="stat-value">{snapshot ? pct(confidence) : '—'}</div>
          <div className="meter" role="meter" aria-valuenow={Math.round(confidence * 100)} aria-valuemin={0} aria-valuemax={100}>
            <div className="meter-fill" style={{ width: pct(confidence) }} />
            <div className="meter-threshold" style={{ left: pct(threshold) }} />
          </div>
          <div className="stat-hint">Alert threshold {pct(threshold)}</div>
        </div>
        <div className="stat">
          <div className="stat-label">Nodes agreeing</div>
          <div className="stat-value">
            {snapshot ? agree : '—'}
            {snapshot && <span className="stat-of"> / {total}</span>}
          </div>
          <div className="stat-hint">{snapshot ? consensusText(snapshot.consensus_status) : '—'}</div>
        </div>
        <div className="stat">
          <div className="stat-label">Detected for</div>
          <div className="stat-value">{detected && detectedSince != null ? duration(snapshot.t - detectedSince) : '—'}</div>
          <div className="stat-hint">{snapshot?.ai_mode ?? ''}</div>
        </div>
      </div>
    </section>
  );
}
