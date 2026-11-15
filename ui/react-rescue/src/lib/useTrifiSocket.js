import { useCallback, useEffect, useRef, useState } from 'react';

export const SCHEMA_VERSION = 1;

/**
 * Backend URL: VITE_WS_URL, else the host serving this page on port 8002
 * (so a tablet opening http://<laptop>:3000 reaches the laptop's backend).
 * Token: ?token=... in the page URL, else VITE_WS_TOKEN.
 */
function backendUrl() {
  const base = import.meta.env.VITE_WS_URL || `ws://${window.location.hostname || 'localhost'}:8002`;
  const token =
    new URLSearchParams(window.location.search).get('token') || import.meta.env.VITE_WS_TOKEN || '';
  const url = token ? `${base}${base.includes('?') ? '&' : '?'}token=${encodeURIComponent(token)}` : base;
  return { url, display: base, hasToken: Boolean(token) };
}

/**
 * Connects to the TRI-FI backend and reconnects with backoff.
 * Also measures the message rate so the System view can show link quality
 * between the backend and this screen.
 */
export function useTrifiSocket() {
  const [status, setStatus] = useState('connecting'); // connecting | online | offline
  const [hello, setHello] = useState(null);
  const [snapshot, setSnapshot] = useState(null);
  const [error, setError] = useState(null);
  const [setup, setSetup] = useState(null);
  const [setupError, setSetupError] = useState(null);
  const [data, setData] = useState(null);
  const [dataError, setDataError] = useState(null);
  const [receivedAt, setReceivedAt] = useState(null);
  const [stats, setStats] = useState({ reconnects: 0, messages: 0, rate: 0, connectedAt: null });
  const wsRef = useRef(null);
  const arrivals = useRef([]);
  const endpoint = useRef(backendUrl());

  useEffect(() => {
    let disposed = false;
    let timer = null;
    let delay = 1000;
    let attempts = 0;

    const connect = () => {
      attempts += 1;
      const ws = new WebSocket(endpoint.current.url);
      wsRef.current = ws;
      setStatus('connecting');
      // A disposed effect's socket (React dev mode mounts twice) must not touch shared state.
      ws.onopen = () => {
        if (disposed) return;
        setStatus('online');
        delay = 1000;
        setStats((s) => ({ ...s, reconnects: attempts - 1, connectedAt: Date.now() }));
      };
      ws.onclose = () => {
        if (wsRef.current === ws) wsRef.current = null;
        if (disposed) return;
        setStatus('offline');
        setStats((s) => ({ ...s, connectedAt: null, rate: 0 }));
        if (!disposed) {
          timer = setTimeout(connect, delay);
          delay = Math.min(delay * 2, 10000);
        }
      };
      ws.onmessage = (event) => {
        if (disposed) return;
        let msg;
        try {
          msg = JSON.parse(event.data);
        } catch {
          return;
        }
        if (msg.schema !== SCHEMA_VERSION) {
          setError(`Backend sends data format ${msg.schema}; this dashboard expects ${SCHEMA_VERSION}. Update one of them.`);
          return;
        }
        if (msg.type === 'hello') {
          setHello(msg);
          setSetup(msg.setup ?? null);
          setData(msg.data ?? null);
          setError(null);
        } else if (msg.type === 'rescue_ai') {
          const now = Date.now();
          arrivals.current.push(now);
          setSnapshot(msg);
          setReceivedAt(now);
        } else if (msg.type === 'setup_state') {
          setSetup(msg);
        } else if (msg.type === 'data_state') {
          setData(msg);
        } else if (msg.type === 'error') {
          if (typeof msg.request === 'string' && msg.request.startsWith('setup_')) setSetupError({ text: msg.error, at: Date.now() });
          else if (typeof msg.request === 'string' && (msg.request.startsWith('label_') || msg.request === 'data_refresh')) setDataError({ text: msg.error, at: Date.now() });
          else setError(msg.error);
        }
      };
    };

    connect();
    // Message rate over the last 5 s, refreshed once a second.
    const rateTimer = setInterval(() => {
      const now = Date.now();
      arrivals.current = arrivals.current.filter((t) => now - t <= 5000);
      setStats((s) => ({ ...s, rate: arrivals.current.length / 5 }));
    }, 1000);

    return () => {
      disposed = true;
      clearTimeout(timer);
      clearInterval(rateTimer);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  /** Send any client message; false when not connected. */
  const send = useCallback((msg) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
      return true;
    }
    return false;
  }, []);

  const setMode = useCallback((mode) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'set_op_mode', mode }));
      return true;
    }
    return false;
  }, []);

  return {
    status,
    hello,
    snapshot,
    error,
    receivedAt,
    stats,
    endpoint: endpoint.current,
    setMode,
    send,
    setup,
    setupError,
    clearSetupError: () => setSetupError(null),
    data,
    dataError,
    clearDataError: () => setDataError(null),
    clearError: () => setError(null),
  };
}
