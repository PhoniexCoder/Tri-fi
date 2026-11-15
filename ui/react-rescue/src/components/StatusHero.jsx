import React from 'react';
import { consensus, duration, nodesAgree, pct } from '../lib/format';
import { Meter } from './charts/Figures';
import Icon from './Icon';

/** The one plain-language answer the operator needs first, plus the next step. */
export function heroState(status, snapshot, ready, missing) {
  if (status !== 'online' || !snapshot)
    return { tone: 'idle', icon: 'wifi', title: status === 'offline' ? 'Backend not connected' : 'Waiting for data',
      next: status === 'offline' ? 'Start the backend with python -m trifi run. The page reconnects by itself.' : 'The first update should arrive within a second.' };
  if (snapshot.ai_detected)
    return { tone: 'alert', icon: 'alert', title: 'Possible survivor detected',
      next: 'Send a team to the marked area and confirm with the rescuer badge. Keep the nodes still.' };
  if (!ready)
    return { tone: 'warn', icon: 'warn', title: 'Getting ready',
      next: `Results are not reliable until this is fixed: ${missing.join(', ').toLowerCase() || 'setup'}.` };
  return { tone: 'info', icon: 'search', title: 'No signs of life detected yet',
    next: 'Keep searching. No detection does not mean the area is clear; move the nodes to cover new ground.' };
}

export default function StatusHero({ state, snapshot, threshold, detectedSince }) {
  const conf = snapshot?.confidence_raw;
  const detected = snapshot?.ai_detected;
  const since = detected && detectedSince != null ? snapshot.t - detectedSince : null;
  const where = consensus(snapshot?.consensus_status);
  return (
    <section className={`hero hero-${state.tone}`} aria-live="polite" aria-labelledby="hero-title">
      <div className="hero-main">
        <div className="hero-icon"><Icon name={state.icon} size={28} /></div>
        <div>
          <h2 id="hero-title" className="hero-title">{state.title}</h2>
          <p className="hero-next">{state.next}</p>
          {snapshot?.rescuer?.confirmed && (
            <p className="hero-chip"><Icon name="badge" size={15} /> Rescue confirmed by the rescuer badge</p>
          )}
        </div>
      </div>

      <div className="hero-figure" title="Fused detection confidence from the active mode. An alert needs this above the threshold and at least one node agreeing.">
        <div className="hero-figure-label">Detection confidence</div>
        <div className="hero-number">{snapshot ? pct(conf) : '—'}</div>
        <Meter value={conf} marker={threshold} tone={detected ? 'alert' : 'accent'}
          label="Detection confidence" valueText={`${pct(conf)}, alert threshold ${pct(threshold)}`} />
        <div className="hero-figure-foot">Alert at {pct(threshold)}</div>
      </div>

      <dl className="hero-facts">
        <div>
          <dt>Nodes agreeing</dt>
          <dd>{snapshot ? `${snapshot.n_detecting} of ${snapshot.nodes?.length ?? 0}` : '—'}</dd>
        </div>
        <div>
          <dt>Where</dt>
          <dd title={where.long}>{snapshot ? where.short : '—'}</dd>
        </div>
        <div>
          <dt>People (estimate)</dt>
          <dd>{detected ? snapshot.survivors : snapshot ? '0' : '—'}</dd>
        </div>
        <div>
          <dt>Detected for</dt>
          <dd>{since != null ? duration(since) : '—'}</dd>
        </div>
      </dl>
      {detected && <p className="sr-only">{nodesAgree(snapshot.n_detecting)}</p>}
    </section>
  );
}
