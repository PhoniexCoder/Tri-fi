import React, { useEffect, useState } from 'react';
import { StatTile } from '../components/charts/Figures';
import Icon, { Status } from '../components/Icon';
import LabelFields, { EMPTY_FIELDS } from '../components/LabelFields';
import LabelTimeline from '../components/LabelTimeline';
import { ACTIVITY_TEXT, bytes, hms, LABEL_COLOR, LABEL_TEXT, parseHms, PLACE_TEXT, sessionDate } from '../lib/recordings';

const EXPORT_CMDS = [
  { cmd: 'python -m trifi export --out training_data', note: 'Two classes: 0 empty, 1 person present' },
  { cmd: 'python -m trifi export --classes 6 --out training_data_6', note: 'The six April classes, to combine with custom_dataset/' },
];

/** Thin two-part bar: share of the recording labelled empty vs person present. */
function Coverage({ row }) {
  const d = row.duration || 0;
  const e = row.labelled?.empty ?? 0;
  const p = row.labelled?.present ?? 0;
  return (
    <div className="coverage" title={`${hms(e)} empty · ${hms(p)} person present · ${hms(Math.max(0, d - e - p))} not labelled`}>
      <div className="coverage-bar">
        {d > 0 && e > 0 && <span style={{ width: `${(e / d) * 100}%`, background: LABEL_COLOR.empty }} />}
        {d > 0 && p > 0 && <span style={{ width: `${(p / d) * 100}%`, background: LABEL_COLOR.present }} />}
      </div>
      <span className="coverage-text">{d ? `${Math.round(((e + p) / d) * 100)}% labelled` : '—'}</span>
    </div>
  );
}

export default function DataView({ ctx }) {
  const { socket, status } = ctx;
  const data = socket.data;
  const [live, setLive] = useState(EMPTY_FIELDS);
  const [selected, setSelected] = useState(null);
  const [editing, setEditing] = useState(null);        // segment id, 'new', or null
  const [form, setForm] = useState({ ...EMPTY_FIELDS, start: '', end: '' });
  const [copied, setCopied] = useState(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const sessions = data?.sessions ?? [];
  const rec = data?.recording;
  const row = sessions.find((s) => s.name === selected) ?? sessions[0] ?? null;

  useEffect(() => { setEditing(null); setConfirmDelete(false); }, [row?.name]);

  if (status !== 'online') return <div className="card empty-boards"><h3 className="card-title">Backend not connected</h3><p className="card-sub">Start it with python -m trifi run.</p></div>;
  if (!data) return <div className="card empty-boards"><h3 className="card-title">Recordings are off during a replay</h3><p className="card-sub">Run the live backend (python -m trifi run) to record and label data.</p></div>;

  const send = (msg) => { socket.clearDataError(); socket.send(msg); };
  const openSeg = rec?.open_segment;
  const recSeconds = rec?.first_t != null && rec?.last_t != null ? rec.last_t - rec.first_t : 0;
  const totals = sessions.reduce((t, s) => ({
    time: t.time + (s.duration || 0), empty: t.empty + (s.labelled?.empty || 0), present: t.present + (s.labelled?.present || 0),
    unlabelled: t.unlabelled + ((s.labelled?.empty || 0) + (s.labelled?.present || 0) === 0 ? 1 : 0),
  }), { time: 0, empty: 0, present: 0, unlabelled: 0 });

  const startEdit = (seg) => {
    const first = row.first_t;
    setEditing(seg ? seg.id : 'new');
    setForm(seg
      ? { label: seg.label, people: seg.people, activity: seg.activity, place: seg.place ?? '', setup: seg.setup, start: hms(seg.start - first), end: seg.end == null ? '' : hms(seg.end - first) }
      : { ...EMPTY_FIELDS, start: '0:00', end: hms(row.duration) });
  };
  const saveEdit = () => {
    const s = parseHms(form.start);
    const e = parseHms(form.end);
    if (s == null || e == null) { socket.clearDataError(); return; }
    const fields = { label: form.label, people: Number(form.people) || 0, activity: form.activity, place: form.place, setup: form.setup,
      start: row.first_t + s, end: row.first_t + e };
    send(editing === 'new' ? { type: 'label_add', session: row.name, ...fields } : { type: 'label_update', session: row.name, id: editing, ...fields });
    setEditing(null);
  };
  const badTimes = editing && (parseHms(form.start) == null || parseHms(form.end) == null || parseHms(form.end) <= parseHms(form.start));

  return (
    <div className="view-col">
      {socket.dataError && (
        <div className="inline-alert inline-bad" role="alert">
          <Icon name="cross" size={16} /><span>{socket.dataError.text}</span>
          <button type="button" className="btn btn-ghost btn-sm" onClick={socket.clearDataError}>Dismiss</button>
        </div>
      )}
      {data.low_disk && <div className="banner banner-warn" role="alert"><Icon name="warn" size={16} /><span>Less than 1 GB free where recordings are saved ({bytes(data.free_bytes)} left). Move old recordings off this laptop.</span></div>}

      <div className="data-top">
        <section className={`card capture ${openSeg ? 'is-labelling' : ''}`} aria-labelledby="cap-title">
          <header className="card-head">
            <div>
              <h3 id="cap-title" className="card-title">Live recording</h3>
              <p className="card-sub">{rec?.active ? `${rec.session} · ${hms(recSeconds)} · ${(rec.packets || 0).toLocaleString()} packets` : 'Not recording. Nothing is saved until you press Start recording.'}</p>
            </div>
            {rec?.active ? <span className="rec-chip"><Icon name="rec" size={12} /> Recording</span> : <Status tone="idle">Off</Status>}
          </header>
          <div className="record-row">
            {rec?.active ? (
              <button type="button" className="btn btn-stop" onClick={() => send({ type: 'record_stop' })}>
                <Icon name="stop" size={15} /> Stop recording
              </button>
            ) : (
              <button type="button" className="btn btn-record" onClick={() => send({ type: 'record_start' })}>
                <Icon name="rec" size={15} /> Start recording
              </button>
            )}
            <span className="field-hint">
              {rec?.active
                ? 'Stopping saves the file and closes any open label. Recordings with no data are discarded.'
                : 'Start once the nodes are placed and the area is set up, so noisy set-up time stays out of your data.'}
            </span>
          </div>
          {openSeg && (
            <div className="labelling-now" aria-live="polite">
              <span className={`seg-dot dot-${openSeg.label}`} aria-hidden="true" />
              <div className="grow">
                <strong>Labelling: {LABEL_TEXT[openSeg.label]}{openSeg.label === 'present' ? `, ${openSeg.people} ${openSeg.people === 1 ? 'person' : 'people'}` : ''}{openSeg.activity ? `, ${ACTIVITY_TEXT[openSeg.activity].toLowerCase()}` : ''}{openSeg.place ? `, ${PLACE_TEXT[openSeg.place].toLowerCase()}` : ''}</strong>
                <span className="muted"> for {hms((rec.last_t ?? openSeg.start) - openSeg.start)}{openSeg.setup ? ` · ${openSeg.setup}` : ''}</span>
              </div>
              <button type="button" className="btn btn-ghost" onClick={() => send({ type: 'label_stop' })}><Icon name="stop" size={14} /> Stop</button>
            </div>
          )}
          <fieldset className="live-label" disabled={!rec?.active}>
            <legend className="sr-only">Label for the live recording</legend>
          <LabelFields value={live} onChange={setLive} activities={data.activities} places={data.places} classes6={data.classes6} idPrefix="live" />
          </fieldset>
          <div className="board-actions">
            <button type="button" className="btn btn-primary" disabled={!rec?.active}
              onClick={() => send({ type: 'label_start', ...live, people: Number(live.people) || 0 })}>
              <Icon name="rec" size={14} /> {openSeg ? 'Switch to this label' : 'Start labelling'}
            </button>
            <span className="field-hint">{rec?.active ? 'Start “Empty” before anyone enters; switch to “Person present” once they are in position.' : 'Start a recording to label it.'}</span>
          </div>
        </section>

        <section className="tiles data-tiles" aria-label="Collected data">
          <StatTile label="Recordings" icon={<Icon name="database" size={15} />} value={String(sessions.length)} detail={`${totals.unlabelled} not labelled yet`} />
          <StatTile label="Recorded time" icon={<Icon name="clock" size={15} />} value={hms(totals.time)} detail="Across all recordings" />
          <StatTile label="Labelled empty" icon={<span className="seg-dot dot-empty" />} value={hms(totals.empty)} detail="Class 0 in exports" />
          <StatTile label="Labelled person present" icon={<span className="seg-dot dot-present" />} value={hms(totals.present)} detail="Class 1 in exports" />
          <StatTile label="Free disk" icon={<Icon name="download" size={15} />} value={bytes(data.free_bytes)} detail="About 150 MB per hour of recording" tone={data.low_disk ? 'warn' : null} />
        </section>
      </div>

      <section className="card" aria-labelledby="rec-title">
        <header className="card-head">
          <div>
            <h3 id="rec-title" className="card-title">All recordings</h3>
            <p className="card-sub">Newest first. Pick one to see and edit its labels. Saved in <code className="path">{data.dir}</code></p>
          </div>
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => send({ type: 'data_refresh' })}><Icon name="replay" size={14} /> Refresh</button>
        </header>
        <div className="table-wrap" tabIndex={0}>
          <table className="data-table rec-table">
            <thead><tr><th scope="col">Recording</th><th scope="col" className="num">Length</th><th scope="col" className="num">Size</th><th scope="col">Labelled</th><th scope="col" className="num">Ranges</th></tr></thead>
            <tbody>
              {sessions.map((s) => (
                <tr key={s.name} className={row?.name === s.name ? 'selected' : ''} onClick={() => setSelected(s.name)}>
                  <th scope="row">
                    <button type="button" className="link-btn" onClick={() => setSelected(s.name)} aria-pressed={row?.name === s.name}>{sessionDate(s.name)}</button>
                    {s.live && <span className="rec-chip rec-chip-sm"><Icon name="rec" size={10} /> live</span>}
                  </th>
                  <td className="num">{hms(s.duration)}</td>
                  <td className="num">{bytes(s.size)}</td>
                  <td><Coverage row={s} /></td>
                  <td className="num">{s.segments.length}</td>
                </tr>
              ))}
              {!sessions.length && <tr><td colSpan={5} className="muted">No recordings yet. Press Record to start one.</td></tr>}
            </tbody>
          </table>
        </div>
      </section>

      {row && (
        <section className="card" aria-labelledby="detail-title">
          <header className="card-head">
            <div>
              <h3 id="detail-title" className="card-title">{sessionDate(row.name)}</h3>
              <p className="card-sub"><code>{row.name}</code> · {hms(row.duration)} · {(row.packets ?? 0).toLocaleString()} packets from {row.sources ?? '—'} {row.sources === 1 ? 'device' : 'devices'}</p>
            </div>
            <div className="btn-group">
              <button type="button" className="btn btn-primary btn-sm" onClick={() => startEdit(null)} disabled={!row.duration}><Icon name="edit" size={14} /> Add a labelled range</button>
              <button type="button" className="btn btn-ghost btn-sm" onClick={() => setConfirmDelete(true)} disabled={row.live}
                title={row.live ? 'Stop the recording before deleting it' : 'Delete this recording and its labels'}>
                <Icon name="trash" size={14} /> Delete recording
              </button>
            </div>
          </header>
          {confirmDelete && (
            <div className="inline-alert inline-bad confirm" role="alertdialog" aria-labelledby="del-q">
              <Icon name="warn" size={16} />
              <span id="del-q">Delete <b>{sessionDate(row.name)}</b> ({hms(row.duration)}, {row.segments.length} labelled {row.segments.length === 1 ? 'range' : 'ranges'})? The file and its labels are removed from this laptop. This can’t be undone.</span>
              <button type="button" className="btn btn-danger btn-sm" onClick={() => { send({ type: 'recording_delete', session: row.name }); setConfirmDelete(false); setSelected(null); }}>Delete</button>
              <button type="button" className="btn btn-ghost btn-sm" onClick={() => setConfirmDelete(false)}>Keep it</button>
            </div>
          )}
          <LabelTimeline row={row} selectedId={editing} onSelect={(id) => startEdit(row.segments.find((s) => s.id === id))} />
          <ul className="map-legend"><li><span className="seg-dot dot-empty" /> Empty</li><li><span className="seg-dot dot-present" /> Person present</li><li><span className="lg-unlabelled" /> Not labelled</li></ul>

          {editing && (
            <div className="segment-editor">
              <h4>{editing === 'new' ? 'New labelled range' : 'Edit labelled range'}</h4>
              <div className="row-fields">
                <label className="field"><span className="field-label">From (m:ss)</span><input className="input input-narrow" value={form.start} onChange={(e) => setForm({ ...form, start: e.target.value })} /></label>
                <label className="field"><span className="field-label">To (m:ss)</span><input className="input input-narrow" value={form.end} onChange={(e) => setForm({ ...form, end: e.target.value })} /></label>
              </div>
              <LabelFields value={form} onChange={(v) => setForm({ ...form, ...v })} activities={data.activities} places={data.places} classes6={data.classes6} idPrefix="edit" />
              {badTimes && <p className="field-warn">Times are minutes:seconds from the start of the recording, and “To” must be after “From”.</p>}
              <div className="board-actions">
                <button type="button" className="btn btn-primary" onClick={saveEdit} disabled={badTimes}>Save</button>
                <button type="button" className="btn btn-ghost" onClick={() => setEditing(null)}>Cancel</button>
              </div>
            </div>
          )}

          <div className="table-wrap">
            <table className="data-table">
              <thead><tr><th scope="col" className="num">From</th><th scope="col" className="num">To</th><th scope="col">Label</th><th scope="col" className="num">People</th><th scope="col">Activity</th><th scope="col">Where</th><th scope="col" className="num" title="Class in a six-class export">Class</th><th scope="col">Setup</th><th scope="col"><span className="sr-only">Actions</span></th></tr></thead>
              <tbody>
                {row.segments.map((s) => (
                  <tr key={s.id} className={editing === s.id ? 'selected' : ''}>
                    <td className="num">{hms(s.start - row.first_t)}</td>
                    <td className="num">{s.end == null ? 'now' : hms(s.end - row.first_t)}</td>
                    <td><span className={`seg-dot dot-${s.label}`} aria-hidden="true" /> {LABEL_TEXT[s.label]}</td>
                    <td className="num">{s.people}</td>
                    <td>{ACTIVITY_TEXT[s.activity]}</td>
                    <td>{PLACE_TEXT[s.place ?? '']}</td>
                    <td className="num" title={s.class6 == null ? 'Skipped in a six-class export' : data.classes6?.[s.class6]}>{s.class6 ?? '—'}</td>
                    <td className="wrap-cell">{s.setup || '—'}</td>
                    <td className="actions">
                      <button type="button" className="btn btn-ghost btn-sm" onClick={() => startEdit(s)} disabled={s.end == null}>Edit</button>
                      <button type="button" className="btn btn-ghost btn-sm" onClick={() => send({ type: 'label_delete', session: row.name, id: s.id })} disabled={s.end == null}>Delete</button>
                    </td>
                  </tr>
                ))}
                {!row.segments.length && <tr><td colSpan={9} className="muted">No labels yet. Add a range above, or label live while recording.</td></tr>}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="card" aria-labelledby="export-title">
        <header className="card-head">
          <div>
            <h3 id="export-title" className="card-title">Use the labels for training</h3>
            <p className="card-sub">Cuts every labelled range into one file per node (64 subcarriers × frames), named like custom_dataset/ so the training script reads them, plus manifest.csv with people, activity and setup notes.</p>
          </div>
        </header>
        <div className="col-cmds">
          {EXPORT_CMDS.map(({ cmd, note }) => (
            <div className="cmd" key={cmd}>
              <div className="grow"><code>{cmd}</code><span className="cmd-note">{note}</span></div>
              <button type="button" className="btn btn-ghost btn-sm" onClick={async () => { try { await navigator.clipboard.writeText(cmd); setCopied(cmd); setTimeout(() => setCopied(null), 1500); } catch { /* clipboard blocked */ } }}>
                <Icon name={copied === cmd ? 'check' : 'copy'} size={14} /> {copied === cmd ? 'Copied' : 'Copy'}
              </button>
            </div>
          ))}
        </div>
        <p className="note"><Icon name="info" size={14} /><span>Train and test on different recordings: keep some sessions out of training so accuracy numbers mean something.</span></p>
      </section>
    </div>
  );
}
