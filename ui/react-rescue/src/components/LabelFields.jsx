import React from 'react';
import { ACTIVITY_TEXT, LABEL_TEXT, PLACE_TEXT, sixClass } from '../lib/recordings';

export const EMPTY_FIELDS = { label: 'empty', people: 0, activity: '', place: '', setup: '' };

/** Label, people, activity and setup notes: shared by live labelling and range editing. */
export default function LabelFields({ value, onChange, activities = ['still', 'breathing', 'moving'], places = ['centre', 'perimeter'], classes6 = {}, idPrefix }) {
  const set = (patch) => onChange({ ...value, ...patch });
  const setLabel = (label) => set({ label, people: label === 'empty' ? 0 : Math.max(1, value.people || 1), ...(label === 'empty' && { activity: '', place: '' }) });
  const six = sixClass(value);
  return (
    <div className="label-fields">
      <div className="field">
        <span className="field-label">What is in the area</span>
        <div className="segmented" role="radiogroup" aria-label="Label">
          {['empty', 'present'].map((l) => (
            <button key={l} type="button" role="radio" aria-checked={value.label === l}
              className={`seg ${value.label === l ? 'on' : ''}`} onClick={() => setLabel(l)}>
              <span className={`seg-dot dot-${l}`} aria-hidden="true" />{LABEL_TEXT[l]}
            </button>
          ))}
        </div>
      </div>
      <label className="field">
        <span className="field-label">People</span>
        <input id={`${idPrefix}-people`} className="input input-narrow" type="number" min={value.label === 'empty' ? 0 : 1} max={20}
          value={value.people} disabled={value.label === 'empty'}
          onChange={(e) => set({ people: e.target.value === '' ? '' : Number(e.target.value) })} />
      </label>
      <div className="field">
        <span className="field-label">Activity (optional)</span>
        <div className="segmented" role="radiogroup" aria-label="Activity">
          {['', ...activities].map((a) => (
            <button key={a || 'none'} type="button" role="radio" aria-checked={value.activity === a}
              className={`seg ${value.activity === a ? 'on' : ''}`} onClick={() => set({ activity: a })}
              disabled={value.label === 'empty' && a !== ''}>{a ? ACTIVITY_TEXT[a] : 'Not set'}</button>
          ))}
        </div>
      </div>
      <div className="field">
        <span className="field-label">Where (optional)</span>
        <div className="segmented" role="radiogroup" aria-label="Where in the area">
          {['', ...places].map((p) => (
            <button key={p || 'none'} type="button" role="radio" aria-checked={value.place === p}
              className={`seg ${value.place === p ? 'on' : ''}`} onClick={() => set({ place: p })}
              disabled={value.label === 'empty' && p !== ''}>{p ? PLACE_TEXT[p] : 'Not set'}</button>
          ))}
        </div>
      </div>
      <p className={`class-preview ${six.cls == null ? 'is-missing' : ''}`} aria-live="polite">
        {six.cls == null
          ? <>Six-class export: <b>skipped</b>, {six.why}</>
          : <>Six-class export: <b>class {six.cls}</b>, {classes6[six.cls] ?? ''}</>}
      </p>
      <label className="field field-wide">
        <span className="field-label">Setup notes</span>
        <input className="input" maxLength={200} value={value.setup} placeholder="e.g. behind brick wall, 2 m from node 1, door closed"
          onChange={(e) => set({ setup: e.target.value })} />
      </label>
    </div>
  );
}
