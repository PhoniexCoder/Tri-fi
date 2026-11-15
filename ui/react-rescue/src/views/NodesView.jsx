import React, { useMemo } from 'react';
import NodeCard from '../components/NodeCard';
import { Status } from '../components/Icon';
import { compact, linkState, num, pct } from '../lib/format';
import { nodeColor, nodeRecords } from '../lib/nodes';
import { nodeValues, recent } from '../lib/series';

/** One detailed card per node, then a comparison table of every field. */
export default function NodesView({ ctx }) {
  const { snapshot, hello, history, nodeThreshold } = ctx;
  const nodeIds = hello?.node_ids ?? [];
  const records = nodeRecords(snapshot, nodeIds);
  const win = useMemo(() => recent(history.samples, 60), [history.version]);
  const positions = hello?.positions ?? {};

  return (
    <div className="view-col">
      <div className="node-grid">
        {records.map((rec) => (
          <NodeCard key={rec.id} id={rec.id} rec={rec} color={nodeColor(nodeIds, rec.id)} threshold={nodeThreshold}
            position={positions[rec.id]}
            spark={{ ml: nodeValues(win, rec.id, 'ml'), rate: nodeValues(win, rec.id, 'rate') }} />
        ))}
        {!records.length && <p className="card muted">No nodes yet. They appear when the backend connects.</p>}
      </div>

      <section className="card" aria-labelledby="nodes-table-title">
        <header className="card-head">
          <div>
            <h3 id="nodes-table-title" className="card-title">All nodes side by side</h3>
            <p className="card-sub">Latest values from the backend</p>
          </div>
        </header>
        <div className="table-wrap" tabIndex={0}>
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Node</th>
                <th scope="col">Link</th>
                <th scope="col" className="num">Model</th>
                <th scope="col">Detecting</th>
                <th scope="col" className="num">Frames/s</th>
                <th scope="col" className="num">Loss</th>
                <th scope="col" className="num">Signal</th>
                <th scope="col" className="num">Frames</th>
                <th scope="col" className="num">Skipped runs</th>
                <th scope="col">Calibration</th>
                <th scope="col" className="num">Breathing</th>
                <th scope="col" className="num">Motion</th>
                <th scope="col">Legacy</th>
                <th scope="col" className="num">Position</th>
              </tr>
            </thead>
            <tbody>
              {records.map(({ id, node, link, cal }) => {
                const h = linkState(link);
                const pos = positions[id];
                return (
                  <tr key={id}>
                    <th scope="row"><span className="swatch" style={{ background: nodeColor(nodeIds, id) }} />Node {id}</th>
                    <td><Status tone={h.tone}>{h.label}</Status></td>
                    <td className="num">{pct(node?.csi_ml_prob)}</td>
                    <td>{node?.csi_detected ? <Status tone="alert">Yes</Status> : 'No'}</td>
                    <td className="num">{num(link?.rate_hz, 1)}</td>
                    <td className="num">{pct(link?.loss, 1)}</td>
                    <td className="num">{link?.rssi ?? node?.rssi ?? '—'} dBm</td>
                    <td className="num">{compact(link?.frames)}</td>
                    <td className="num">{compact(link?.dropped_inference)}</td>
                    <td>{cal?.done ? 'Complete' : cal ? pct(cal.progress) : '—'}</td>
                    <td className="num">{node?.breathing ? `${num(node.breathing, 1)}/min${node.breathing_detected ? ' ✓' : ''}` : '—'}</td>
                    <td className="num">{num(node?.motion, 2)}</td>
                    <td>{node?.active ? 'Active' : 'Inactive'}</td>
                    <td className="num">{pos ? `${pos[0].toFixed(2)}, ${pos[1].toFixed(2)}` : '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
