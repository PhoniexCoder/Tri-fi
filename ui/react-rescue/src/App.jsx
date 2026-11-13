import React, { useEffect, useState } from 'react';
import ConfidenceChart from './components/ConfidenceChart';
import EventLog from './components/EventLog';
import FieldMap from './components/FieldMap';
import NodeList from './components/NodeList';
import Readiness from './components/Readiness';
import StatusHero from './components/StatusHero';
import TopBar from './components/TopBar';
import { readiness } from './lib/readiness';
import { useHistory } from './lib/useHistory';
import { useTrifiSocket } from './lib/useTrifiSocket';

function loadTheme() {
  try {
    return localStorage.getItem('trifi-theme') || 'dark';
  } catch {
    return 'dark';
  }
}

export default function App() {
  const { status, hello, snapshot, error, receivedAt, setMode } = useTrifiSocket();
  const { points, events, detectedSince } = useHistory(snapshot, status, hello);
  const [theme, setTheme] = useState(loadTheme);
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem('trifi-theme', theme);
    } catch {
      /* storage unavailable: theme just won't persist */
    }
  }, [theme]);

  const nodeIds = hello?.node_ids ?? [];
  const { items, ready } = readiness(status, hello, snapshot);
  const ageSeconds = receivedAt ? (now - receivedAt) / 1000 : null;

  return (
    <div className="app">
      <TopBar
        status={status}
        hello={hello}
        snapshot={snapshot}
        ageSeconds={ageSeconds}
        onMode={setMode}
        theme={theme}
        onTheme={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
      />

      {error && <div className="banner banner-bad">{error}</div>}
      {status === 'offline' && (
        <div className="banner banner-bad">
          Can't reach the backend. Check it is running (<code>python -m trifi run</code>) and, if the kit
          uses a token, that this page's address ends in <code>?token=…</code>. Retrying automatically.
        </div>
      )}

      <main className="layout">
        <StatusHero snapshot={snapshot} ready={ready} detectedSince={detectedSince} />
        <div className="col col-left">
          <Readiness items={items} ready={ready} />
          <ConfidenceChart points={points} threshold={snapshot?.ui_threshold ?? 0.3} />
          <EventLog events={events} />
        </div>
        <div className="col col-center">
          <FieldMap snapshot={snapshot} hello={hello} />
        </div>
        <div className="col col-right">
          <NodeList snapshot={snapshot} nodeIds={nodeIds} />
        </div>
      </main>
    </div>
  );
}
