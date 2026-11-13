import React from 'react';
import { nodeColor, pct } from '../lib/geometry';

/** Per-node link quality and calibration: is each node actually delivering data? */
export default function LinkHealth({ snapshot, nodeIds }) {
  const links = snapshot?.links ?? {};
  const calibration = snapshot?.calibration ?? {};
  return (
    <>
      <div className="panel-title">Link Health</div>
      <div>
        {nodeIds.map((id) => {
          const link = links[id] ?? {};
          const cal = calibration[id] ?? {};
          const rate = link.rate_hz ?? 0;
          const loss = link.loss ?? 0;
          const state = rate === 0 ? 'down' : rate < 20 || loss > 0.05 ? 'weak' : 'good';
          return (
            <div className={`link-row link-${state}`} key={id}>
              <div className="link-head">
                <span className="node-dot" style={{ background: nodeColor(id) }} />
                <span className="node-name">NODE {id}</span>
                <span className="link-state">{state.toUpperCase()}</span>
              </div>
              <div className="link-metrics">
                <span>{rate.toFixed(0)} Hz</span>
                <span>loss {pct(loss)}</span>
                <span>{link.rssi ?? '--'} dBm</span>
              </div>
              {cal.done ? (
                <div className="calib-done">Calibrated</div>
              ) : (
                <div className="calib-track" title="Empty-room calibration — keep the area clear">
                  <div className="calib-fill" style={{ width: pct(cal.progress ?? 0) }} />
                  <span>Calibrating {pct(cal.progress ?? 0, 0)}</span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </>
  );
}
