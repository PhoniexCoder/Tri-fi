import React from 'react';

export default function RescuerCard({ rescuer }) {
  const r = rescuer ?? { active: false, confirmed: false };
  return (
    <>
      <div className="panel-title">Rescuer Badge</div>
      <div id="rescuer-card" className={r.active ? 'active' : ''}>
        <div className="status-row">
          <span className="lbl">State:</span>
          <span className={`val ${r.active ? 'online' : 'offline'}`}>
            {r.active ? 'BADGE ONLINE' : 'NO BADGE'}
          </span>
        </div>
        {r.confirmed && (
          <div className="status-row confirmed">
            <span className="lbl">Status:</span>
            <span className="val" style={{ color: 'var(--node3)' }}>
              RESCUE CONFIRMED
            </span>
          </div>
        )}
      </div>
    </>
  );
}
