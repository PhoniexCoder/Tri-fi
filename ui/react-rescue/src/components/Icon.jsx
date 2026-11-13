import React from 'react';

const PATHS = {
  check: 'M5 12.5l4.5 4.5L19 7.5',
  warn: 'M12 4l9 16H3L12 4zm0 6v4.5m0 3v.01',
  cross: 'M6 6l12 12M18 6L6 18',
  dot: 'M12 12m-4 0a4 4 0 1 0 8 0a4 4 0 1 0 -8 0',
  info: 'M12 8v.01M12 11v6M12 3a9 9 0 1 0 0 18a9 9 0 0 0 0-18z',
  sun: 'M12 7a5 5 0 1 0 0 10a5 5 0 0 0 0-10zM12 1v2M12 21v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M1 12h2M21 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4',
  moon: 'M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z',
  pulse: 'M3 12h4l2-6 4 12 2-6h6',
};

/** Small stroke icon; pair with text so meaning never depends on colour alone. */
export default function Icon({ name, size = 18, className = '' }) {
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
      aria-hidden="true"
    >
      <path d={PATHS[name]} />
    </svg>
  );
}

export const TONE_ICON = { ok: 'check', warn: 'warn', bad: 'cross', alert: 'warn', info: 'info' };
