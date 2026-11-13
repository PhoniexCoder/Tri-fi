import { useCallback, useEffect, useRef, useState } from 'react';

export const SCHEMA_VERSION = 1;

/**
 * Backend URL: VITE_WS_URL, else the host serving this page on port 8002
 * (so a tablet opening http://<laptop>:3000 reaches the laptop's backend).
 * Token: ?token=... in the page URL, else VITE_WS_TOKEN.
 */
function backendUrl() {
  const base =
    import.meta.env.VITE_WS_URL || `ws://${window.location.hostname || 'localhost'}:8002`;
  const token =
    new URLSearchParams(window.location.search).get('token') ||
    import.meta.env.VITE_WS_TOKEN ||
    '';
  if (!token) return base;
  return `${base}${base.includes('?') ? '&' : '?'}token=${encodeURIComponent(token)}`;
}

/** Connects to the TRI-FI backend; reconnects with backoff. */
export function useTrifiSocket() {
  const [status, setStatus] = useState('connecting'); // connecting | online | offline
  const [hello, setHello] = useState(null);
  const [snapshot, setSnapshot] = useState(null);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => {
    let disposed = false;
    let timer = null;
    let delay = 1000;

    const connect = () => {
      const ws = new WebSocket(backendUrl());
      wsRef.current = ws;
      setStatus('connecting');
      ws.onopen = () => {
        setStatus('online');
        delay = 1000;
      };
      ws.onclose = () => {
        wsRef.current = null;
        setStatus('offline');
        if (!disposed) {
          timer = setTimeout(connect, delay);
          delay = Math.min(delay * 2, 10000);
        }
      };
      ws.onmessage = (event) => {
        let msg;
        try {
          msg = JSON.parse(event.data);
        } catch {
          return;
        }
        if (msg.schema !== SCHEMA_VERSION) {
          setError(
            `Backend sends schema ${msg.schema}; this dashboard expects ${SCHEMA_VERSION}. Update one of them.`,
          );
          return;
        }
        if (msg.type === 'hello') {
          setHello(msg);
          setError(null);
        } else if (msg.type === 'rescue_ai') {
          setSnapshot(msg);
        } else if (msg.type === 'error') {
          setError(msg.error);
        }
      };
    };

    connect();
    return () => {
      disposed = true;
      clearTimeout(timer);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const setMode = useCallback((mode) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'set_op_mode', mode }));
    }
  }, []);

  return { status, hello, snapshot, error, setMode };
}
