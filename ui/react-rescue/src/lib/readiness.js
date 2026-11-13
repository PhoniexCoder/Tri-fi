import { linkState, pct } from './format';

/**
 * Checklist of what must be true before search results mean anything.
 * ready = connected, every node sending data, every node calibrated.
 */
export function readiness(status, hello, snapshot) {
  const nodeIds = hello?.node_ids ?? [];
  const links = snapshot?.links ?? {};
  const cal = snapshot?.calibration ?? {};
  const items = [];

  const connected = status === 'online';
  items.push({
    key: 'backend',
    tone: connected ? 'ok' : 'bad',
    label: 'Backend',
    detail: connected ? (hello?.live === false ? 'Connected (replay)' : 'Connected') : 'Not connected',
  });

  const model = snapshot?.model;
  items.push({
    key: 'model',
    tone: !snapshot ? 'idle' : model?.loaded ? 'ok' : 'warn',
    label: 'Detection model',
    detail: !snapshot ? 'Unknown' : model?.loaded ? `Loaded (${model.classes} classes)` : 'Not loaded — legacy RSSI only',
  });

  const up = nodeIds.filter((id) => linkState(links[id]).key !== 'down');
  const weak = nodeIds.filter((id) => linkState(links[id]).key === 'weak');
  items.push({
    key: 'nodes',
    tone: !snapshot ? 'idle' : up.length === nodeIds.length && nodeIds.length ? (weak.length ? 'warn' : 'ok') : up.length ? 'warn' : 'bad',
    label: 'Nodes sending data',
    detail: snapshot
      ? `${up.length} of ${nodeIds.length}${weak.length ? ` (weak: node ${weak.join(', ')})` : ''}`
      : '—',
  });

  const calibrated = nodeIds.filter((id) => cal[id]?.done);
  const progress = nodeIds.length
    ? nodeIds.reduce((s, id) => s + (cal[id]?.progress ?? 0), 0) / nodeIds.length
    : 0;
  items.push({
    key: 'calibration',
    tone: !snapshot ? 'idle' : calibrated.length === nodeIds.length && nodeIds.length ? 'ok' : 'warn',
    label: 'Empty-area calibration',
    detail: !snapshot
      ? '—'
      : calibrated.length === nodeIds.length
        ? 'Complete'
        : `In progress ${pct(progress)} — keep the area clear`,
  });

  const r = snapshot?.rescuer;
  items.push({
    key: 'badge',
    tone: r?.confirmed ? 'ok' : r?.active ? 'ok' : 'idle',
    label: 'Rescuer badge',
    detail: r?.confirmed ? 'Rescue confirmed' : r?.active ? 'Online' : 'Not seen (optional)',
  });

  const ready =
    connected &&
    nodeIds.length > 0 &&
    up.length === nodeIds.length &&
    calibrated.length === nodeIds.length;
  return { items, ready };
}
