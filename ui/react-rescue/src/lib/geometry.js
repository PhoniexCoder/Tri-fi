/** Map backend unit coordinates (0..1) onto the 500x500 field SVG. */
export function posToSvg(px, py) {
  return { x: 80 + px * 340, y: 80 + py * 360 };
}

const NODE_COLORS = ['var(--node1)', 'var(--node2)', 'var(--node3)', 'var(--orange)', 'var(--accent2)'];

export function nodeColor(nodeId) {
  return NODE_COLORS[(nodeId - 1) % NODE_COLORS.length];
}

export const pct = (v, digits = 1) => `${((v ?? 0) * 100).toFixed(digits)}%`;
