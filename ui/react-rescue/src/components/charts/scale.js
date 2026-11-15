import { useLayoutEffect, useRef, useState } from 'react';

/** Linear scale. */
export const linear = (d0, d1, r0, r1) => (v) => (d1 === d0 ? (r0 + r1) / 2 : r0 + ((v - d0) / (d1 - d0)) * (r1 - r0));

/** Clean tick values (1/2/5 steps) covering [min, max]. */
export function niceTicks(min, max, count = 4) {
  if (!(max > min)) return [min];
  const raw = (max - min) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

/** Extent of finite values across series, padded and snapped to ticks. */
export function autoDomain(series, { min, max, pad = 0.1 } = {}) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of series) for (const v of s.values) if (v != null && Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (!Number.isFinite(lo)) return [min ?? 0, max ?? 1];
  if (hi === lo) { hi += 1; lo -= 1; }
  const span = hi - lo;
  lo = min ?? lo - span * pad;
  hi = max ?? hi + span * pad;
  const ticks = niceTicks(lo, hi);
  return [min ?? Math.min(lo, ticks[0]), max ?? Math.max(hi, ticks[ticks.length - 1])];
}

/** Index of the time closest to t (times sorted ascending). */
export function nearestIndex(times, t) {
  let lo = 0;
  let hi = times.length - 1;
  if (hi < 0) return -1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < t) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(times[lo - 1] - t) < Math.abs(times[lo] - t)) return lo - 1;
  return lo;
}

/** SVG path through points, breaking at null values; optional step shape. */
export function linePath(times, values, x, y, step = false) {
  let d = '';
  let pen = false;
  let prevY = 0;
  for (let i = 0; i < times.length; i++) {
    const v = values[i];
    if (v == null || !Number.isFinite(v)) { pen = false; continue; }
    const px = x(times[i]).toFixed(1);
    const py = y(v).toFixed(1);
    if (!pen) d += `M${px},${py}`;
    else if (step) d += `H${px}V${py}`;
    else d += `L${px},${py}`;
    pen = true;
    prevY = py;
  }
  return { d, prevY };
}

/** Width of an element, tracked with ResizeObserver, so text in SVG never scales. */
export function useWidth(initial = 600) {
  const ref = useRef(null);
  const [width, setWidth] = useState(initial);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    setWidth(el.clientWidth || initial);
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(200, Math.floor(entry.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}
