import React, { useEffect, useRef, useState } from 'react';
import Icon from './components/Icon';
import Nav, { VIEWS } from './components/Nav';
import { heroState } from './components/StatusHero';
import TopBar, { connection } from './components/TopBar';
import { linkState } from './lib/format';
import { readiness } from './lib/readiness';
import { useHistory } from './lib/useHistory';
import { useTrifiSocket } from './lib/useTrifiSocket';
import EventsView from './views/EventsView';
import MapView from './views/MapView';
import NodesView from './views/NodesView';
import OverviewView from './views/OverviewView';
import SetupView from './views/SetupView';
import SignalsView from './views/SignalsView';
import SystemView from './views/SystemView';

const PAGES = { overview: OverviewView, map: MapView, nodes: NodesView, signals: SignalsView, events: EventsView, setup: SetupView, system: SystemView };
const STALE_UPDATES = 5; // missed broadcasts before the screen warns that data stopped

function storedTheme() {
  try {
    return localStorage.getItem('trifi-theme') || 'dark';
  } catch {
    return 'dark';
  }
}
function storedSsid() {
  try {
    return localStorage.getItem('trifi-setup-ssid') || '';
  } catch {
    return '';
  }
}
const viewFromHash = () => {
  const id = window.location.hash.replace(/^#\/?/, '');
  return PAGES[id] ? id : 'overview';
};

export default function App() {
  const socket = useTrifiSocket();
  const { status, hello, snapshot, error, receivedAt } = socket;
  const history = useHistory({ snapshot, status, hello, error });
  const [theme, setTheme] = useState(storedTheme);
  const [view, setView] = useState(viewFromHash);
  const [now, setNow] = useState(Date.now());
  // Wi-Fi form lives here so it survives switching views; the password is never persisted.
  const [setupForm, setSetupForm] = useState(() => ({ ssid: storedSsid(), password: '', targetIp: '' }));
  const [newBoards, setNewBoards] = useState([]);
  const seenPorts = useRef(null);

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 500);
    const onHash = () => setView(viewFromHash());
    window.addEventListener('hashchange', onHash);
    return () => { clearInterval(timer); window.removeEventListener('hashchange', onHash); };
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('trifi-theme', theme); } catch { /* theme just won't persist */ }
  }, [theme]);

  useEffect(() => {
    try { localStorage.setItem('trifi-setup-ssid', setupForm.ssid); } catch { /* not remembered */ }
  }, [setupForm.ssid]);

  // Auto-detect: announce boards plugged in (or removed) while the dashboard is open.
  const portList = (socket.setup?.ports ?? []).map((p) => p.device);
  const portKey = portList.join(',');
  useEffect(() => {
    if (!socket.setup) return;
    const now = new Set(portList);
    if (seenPorts.current) {
      const added = portList.filter((d) => !seenPorts.current.has(d));
      const removed = [...seenPorts.current].filter((d) => !now.has(d));
      for (const d of added) history.addEvent('info', `ESP32 board connected on ${d}`);
      for (const d of removed) history.addEvent('info', `ESP32 board on ${d} unplugged`);
      setNewBoards((list) => [...list.filter((d) => now.has(d)), ...added.filter((d) => !list.includes(d))]);
    } else {
      setNewBoards(portList);
    }
    seenPorts.current = now;
  }, [portKey, Boolean(socket.setup)]);

  useEffect(() => {
    if (view === 'setup') setNewBoards([]);
  }, [view, portKey]);

  const go = (id) => {
    window.location.hash = `/${id}`;
    setView(id);
    window.scrollTo({ top: 0 });
  };

  useEffect(() => {
    const onKey = (e) => {
      if (e.altKey || e.ctrlKey || e.metaKey || /input|textarea|select/i.test(e.target.tagName)) return;
      const i = Number(e.key) - 1;
      if (i >= 0 && i < VIEWS.length) go(VIEWS[i].id);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const broadcastHz = hello?.broadcast_hz ?? 5;
  const staleAfter = Math.max(2, STALE_UPDATES / broadcastHz);
  const ageSeconds = receivedAt ? (now - receivedAt) / 1000 : null;
  const conn = connection(status, hello, snapshot, ageSeconds, staleAfter);
  const { items, ready, missing } = readiness(status, hello, snapshot);
  const hero = heroState(status, snapshot, ready, missing);
  const detected = Boolean(snapshot?.ai_detected) && status === 'online';

  useEffect(() => {
    document.title = detected ? '⚠ Possible survivor · TRI-FI' : 'TRI-FI · Survivor search console';
  }, [detected]);

  const nodeIds = hello?.node_ids ?? [];
  const unhealthy = snapshot ? nodeIds.filter((id) => linkState(snapshot.links?.[id]).key !== 'good').length : 0;
  const badges = {
    ...(detected && { overview: { tone: 'alert', text: 'Alert' } }),
    ...(unhealthy && { nodes: { tone: 'warn', text: String(unhealthy) } }),
    ...(portList.length && { setup: { tone: 'info', text: String(portList.length) } }),
  };

  const ctx = {
    status, hello, snapshot, history, socket, conn, ageSeconds, now, ready, items, hero,
    threshold: snapshot?.ui_threshold ?? hello?.ui_threshold ?? 0.3,
    nodeThreshold: hello?.node_threshold ?? 0.45,
    setupForm, setSetupForm,
  };
  const Page = PAGES[view];
  const current = VIEWS.find((v) => v.id === view);

  return (
    <div className={`app ${detected ? 'is-alert' : ''}`}>
      <a className="skip" href="#main">Skip to content</a>
      <TopBar conn={conn} hello={hello} snapshot={snapshot} onMode={socket.setMode} theme={theme}
        onTheme={() => setTheme(theme === 'dark' ? 'light' : 'dark')} />
      <div className="shell">
        <Nav view={view} onView={go} badges={badges} />
        <main id="main" className="main" tabIndex={-1}>
          {error && (
            <div className="banner banner-bad" role="alert">
              <Icon name="cross" size={16} /> <span>{error}</span>
              <button type="button" className="btn btn-ghost btn-sm push-right" onClick={socket.clearError}>Dismiss</button>
            </div>
          )}
          {status === 'offline' && (
            <div className="banner banner-bad" role="alert">
              <Icon name="wifiOff" size={16} />
              <span>Can't reach the backend. Check it is running (<code>python -m trifi run</code>) and, if the kit uses a token, that this page's address ends in <code>?token=…</code>. Retrying automatically.</span>
            </div>
          )}
          {status === 'online' && conn.tone === 'warn' && (
            <div className="banner banner-warn" role="alert">
              <Icon name="clock" size={16} />
              <span>No update for {ageSeconds.toFixed(0)} s. The values below are frozen at the last update; don't rely on them until updates resume.</span>
            </div>
          )}
          {detected && view !== 'overview' && (
            <button type="button" className="banner banner-alert" onClick={() => go('overview')}>
              <Icon name="alert" size={16} /> <span><strong>Possible survivor detected.</strong> Open the overview.</span>
            </button>
          )}
          {newBoards.length > 0 && view !== 'setup' && (
            <div className="banner banner-info" role="status">
              <Icon name="usb" size={16} />
              <span><strong>ESP32 connected on {newBoards.join(', ')}.</strong> Install TRI-FI firmware with just the Wi-Fi name and password.</span>
              <span className="push-right btn-group">
                <button type="button" className="btn btn-primary btn-sm" onClick={() => { setNewBoards([]); go('setup'); }}>Set it up</button>
                <button type="button" className="btn btn-ghost btn-sm" onClick={() => setNewBoards([])}>Not now</button>
              </span>
            </div>
          )}
          <div className="view-head">
            <h1>{current.label}</h1>
            <p>{current.hint}</p>
          </div>
          <div id={`view-${view}`} role="tabpanel" className={conn.tone === 'warn' ? 'is-stale' : ''}>
            <Page ctx={ctx} go={go} />
          </div>
        </main>
      </div>
    </div>
  );
}
