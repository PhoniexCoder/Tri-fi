/**
 * Node identity colour. Colour follows the node (its position in the
 * configured node list), never its rank, so a node keeps its colour when
 * others drop out. Slots 1-6 of the validated categorical palette are used;
 * slot 7 (violet) is reserved for fused confidence. Nodes past the sixth
 * share the neutral "other" colour and rely on their number label.
 */
export const NODE_SLOTS = 6;

export function nodeColor(nodeIds, id) {
  const i = nodeIds.indexOf(Number(id));
  return i >= 0 && i < NODE_SLOTS ? `var(--series-${i + 1})` : 'var(--series-other)';
}

/** Latest per-node record merged from nodes[], links{} and calibration{}. */
export function nodeRecords(snapshot, nodeIds) {
  const byId = Object.fromEntries((snapshot?.nodes ?? []).map((n) => [n.node_id, n]));
  return nodeIds.map((id) => ({
    id,
    node: byId[id] ?? null,
    link: snapshot?.links?.[id] ?? null,
    cal: snapshot?.calibration?.[id] ?? null,
  }));
}
