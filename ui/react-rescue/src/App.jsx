import React from 'react';
import FieldMap from './components/FieldMap';
import Header from './components/Header';
import LinkHealth from './components/LinkHealth';
import NodeDetails from './components/NodeDetails';
import RescuerCard from './components/RescuerCard';
import SituationPanel from './components/SituationPanel';
import { useTrifiSocket } from './lib/useTrifiSocket';

export default function App() {
  const { status, hello, snapshot, error, setMode } = useTrifiSocket();
  const nodeIds = hello?.node_ids ?? [];
  const detected = snapshot?.ai_detected ?? false;

  return (
    <div className="app">
      <Header status={status} hello={hello} snapshot={snapshot} onMode={setMode} />
      {error && <div className="banner-error">{error}</div>}
      {status !== 'online' && (
        <div className="banner-error">
          No connection to the backend. Check that it is running, and that the page URL carries
          ?token=… if the kit uses one.
        </div>
      )}

      <div id="main">
        <div id="panel-left">
          <SituationPanel snapshot={snapshot} />
          <LinkHealth snapshot={snapshot} nodeIds={nodeIds} />
          <RescuerCard rescuer={snapshot?.rescuer} />
        </div>

        <div id="center">
          <div id="alert-banner" style={{ display: detected ? 'block' : 'none' }}>
            ⚠ POSSIBLE SURVIVOR — CONFIRM BY CALL-OUT, ACOUSTIC OR CAMERA
          </div>
          <FieldMap snapshot={snapshot} hello={hello} />
        </div>

        <div id="panel-right">
          <NodeDetails snapshot={snapshot} nodeIds={nodeIds} />
        </div>
      </div>
    </div>
  );
}
