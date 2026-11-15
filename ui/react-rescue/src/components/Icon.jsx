import React from 'react';

// 24x24 stroke icons. Always paired with text, so meaning never rests on an icon or colour alone.
const PATHS = {
  check: 'M5 12.5l4.5 4.5L19 7.5',
  warn: 'M12 4l9 16H3L12 4zm0 6v4.5m0 3v.01',
  cross: 'M6 6l12 12M18 6L6 18',
  info: 'M12 8v.01M12 11v6M12 3a9 9 0 1 0 0 18a9 9 0 0 0 0-18z',
  alert: 'M12 3a9 9 0 1 0 0 18a9 9 0 0 0 0-18zM12 7.5v5.5m0 3.5v.01',
  dot: 'M12 9a3 3 0 1 0 0 6a3 3 0 0 0 0-6z',
  sun: 'M12 7a5 5 0 1 0 0 10a5 5 0 0 0 0-10zM12 1v2M12 21v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M1 12h2M21 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4',
  moon: 'M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z',
  pulse: 'M3 12h4l2-6 4 12 2-6h6',
  home: 'M3 11l9-7 9 7M5 9.5V20h5v-6h4v6h5V9.5',
  nodes: 'M12 5a2 2 0 1 0 0 .01M5 18a2 2 0 1 0 0 .01M19 18a2 2 0 1 0 0 .01M12 7l-6 9.5M12 7l6 9.5M7 18h10',
  chart: 'M4 20V4M4 20h16M8 16l4-5 3 3 5-7',
  map: 'M9 4L3 6.5v13.5l6-2.5 6 2.5 6-2.5V4l-6 2.5L9 4zM9 4v13.5M15 6.5V20',
  list: 'M9 6h11M9 12h11M9 18h11M4.5 6h.01M4.5 12h.01M4.5 18h.01',
  cog: 'M12 9a3 3 0 1 0 0 6a3 3 0 0 0 0-6zM12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1L7 17M17 7l2.1-2.1',
  wifi: 'M2 8.5a15 15 0 0 1 20 0M5 12a10 10 0 0 1 14 0M8.5 15.5a5 5 0 0 1 7 0M12 19h.01',
  wifiOff: 'M3 3l18 18M8.5 15.5a5 5 0 0 1 7 0M5 12a10 10 0 0 1 5-2.6M14.5 9.6A10 10 0 0 1 19 12M2 8.5a15 15 0 0 1 4.3-2.8M12 19h.01',
  clock: 'M12 3a9 9 0 1 0 0 18a9 9 0 0 0 0-18zM12 7v5l3 2',
  user: 'M12 4a4 4 0 1 0 0 8a4 4 0 0 0 0-8zM4 21a8 8 0 0 1 16 0',
  badge: 'M12 2l7 3v6c0 5-3 9-7 11c-4-2-7-6-7-11V5l7-3zM9 12l2 2 4-4',
  lungs: 'M12 4v8M12 12c-1.5-1-3-1.5-4-1.5C6 10.5 4 13 4 17c0 2 1 3 2.5 3S10 18 12 15M12 12c1.5-1 3-1.5 4-1.5c2 0 4 2.5 4 6.5c0 2-1 3-2.5 3S14 18 12 15',
  download: 'M12 4v11M7 10.5l5 5 5-5M5 20h14',
  pause: 'M8 5v14M16 5v14',
  play: 'M7 5l12 7-12 7V5z',
  table: 'M4 5h16v14H4zM4 10h16M4 15h16M10 5v14',
  layers: 'M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5',
  copy: 'M9 9h11v11H9zM5 15H4V4h11v1',
  search: 'M11 4a7 7 0 1 0 0 14a7 7 0 0 0 0-14zM20 20l-4-4',
  trash: 'M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3',
  replay: 'M4 12a8 8 0 1 0 2.3-5.7M4 4v4h4',
  target: 'M12 3a9 9 0 1 0 0 18a9 9 0 0 0 0-18zM12 8a4 4 0 1 0 0 8a4 4 0 0 0 0-8zM12 11.5v1',
  usb: 'M12 3v13M12 3l-2.5 3h5L12 3zM12 16a2 2 0 1 0 0 4a2 2 0 0 0 0-4zM12 12l-5-2.5V7M12 13.5l5-2.5V8.5M5.5 5.5h3v1.5h-3zM16 6.5a1 1 0 1 0 2 0a1 1 0 0 0-2 0',
  chip: 'M7 7h10v10H7zM10 10h4v4h-4zM9 3v4M15 3v4M9 17v4M15 17v4M3 9h4M3 15h4M17 9h4M17 15h4',
  eye: 'M2 12s3.5-7 10-7s10 7 10 7s-3.5 7-10 7S2 12 2 12zM12 9a3 3 0 1 0 0 6a3 3 0 0 0 0-6z',
  eyeOff: 'M3 3l18 18M10.6 5.1A10 10 0 0 1 12 5c6.5 0 10 7 10 7a17 17 0 0 1-3 3.9M6.6 6.6C3.8 8.4 2 12 2 12s3.5 7 10 7a10 10 0 0 0 5.4-1.6M9.9 9.9a3 3 0 0 0 4.2 4.2',
  bolt: 'M13 2L4 14h7l-1 8 9-12h-7l1-8z',
  database: 'M12 3c4.4 0 8 1.3 8 3s-3.6 3-8 3s-8-1.3-8-3s3.6-3 8-3zM4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3',
  rec: 'M12 7a5 5 0 1 0 0 10a5 5 0 0 0 0-10z',
  stop: 'M7 7h10v10H7z',
  edit: 'M4 20h4L19 9l-4-4L4 16v4zM14 6l4 4',
};

export default function Icon({ name, size = 18, className = '', title }) {
  return (
    <svg
      className={`icon ${className}`}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden={title ? undefined : 'true'}
      role={title ? 'img' : undefined}
    >
      {title && <title>{title}</title>}
      <path d={PATHS[name] ?? PATHS.dot} />
    </svg>
  );
}

/** Status tone -> icon. Tones: ok, warn, bad, alert, info, idle. */
export const TONE_ICON = { ok: 'check', warn: 'warn', bad: 'cross', alert: 'alert', info: 'info', idle: 'dot' };

/** Status chip: icon + label, coloured by tone. */
export function Status({ tone = 'idle', children, title }) {
  return (
    <span className={`status status-${tone}`} title={title}>
      <Icon name={TONE_ICON[tone]} size={14} />
      <span>{children}</span>
    </span>
  );
}
