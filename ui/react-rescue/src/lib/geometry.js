/** Field map size and padding (room for labels and range rings). */
export const MAP = { w: 520, h: 440, pad: 70 };

/** Map backend unit coordinates (0..1) onto the field SVG. */
export function posToSvg(px, py) {
  return {
    x: MAP.pad + px * (MAP.w - 2 * MAP.pad),
    y: MAP.pad + py * (MAP.h - 2 * MAP.pad),
  };
}
