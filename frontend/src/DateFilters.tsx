import { useState } from 'react';
import { CalendarDays, RotateCcw } from 'lucide-react';

export type DateRange = { start: string; end: string };
const storageKey = 'vanpredict.dateRange.v1';

function validDate(value: string) {
  return /^\d{4}-\d{2}-\d{2}$/.test(value) && !Number.isNaN(Date.parse(value)) && new Date(value).toISOString().slice(0, 10) === value;
}

export function readSavedRange(): DateRange | null {
  try {
    const range = JSON.parse(localStorage.getItem(storageKey) ?? 'null');
    return range && validDate(range.start) && validDate(range.end) && range.start <= range.end ? range : null;
  } catch { return null; }
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat('en-CA', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' }).format(new Date(`${value}T00:00:00Z`));
}

export function formatRange(range: DateRange | null) {
  return range ? `${formatDate(range.start)} – ${formatDate(range.end)}` : 'All available dates';
}

export default function DateFilters({ range, onApply }: { range: DateRange | null; onApply: (range: DateRange | null) => void }) {
  const [start, setStart] = useState(range?.start ?? '');
  const [end, setEnd] = useState(range?.end ?? '');
  const [error, setError] = useState('');
  const [storageWarning, setStorageWarning] = useState('');
  const pending = start !== (range?.start ?? '') || end !== (range?.end ?? '');
  function save(next: DateRange | null) {
    setStorageWarning('');
    try {
      if (next) localStorage.setItem(storageKey, JSON.stringify(next));
      else localStorage.removeItem(storageKey);
    } catch { setStorageWarning('This browser cannot save the period between visits.'); }
    onApply(next);
  }
  return <section className="date-filter-card" aria-labelledby="date-filter-title">
    <div className="date-filter-intro"><span className="date-icon"><CalendarDays size={18}/></span><div><h2 id="date-filter-title">Choose a period</h2><p>Explore a moment in the city's history.</p></div></div>
    <form className="date-form" noValidate onSubmit={event => {
      event.preventDefault();
      const values = new FormData(event.currentTarget);
      const start = String(values.get('start') ?? '');
      const end = String(values.get('end') ?? '');
      setStart(start); setEnd(end);
      if (!validDate(start) || !validDate(end)) { setError('Choose a valid start date and end date.'); return; }
      if (start > end) { setError('The end date must be on or after the start date.'); return; }
      setError(''); save({ start, end });
    }}>
      <div className="date-fields"><label htmlFor="start-date">Start date<input id="start-date" name="start" type="date" value={start} aria-invalid={!!error} aria-describedby={error ? 'date-error' : 'date-help'} onInput={event => { setStart(event.currentTarget.value); setError(''); }}/></label><span className="date-to" aria-hidden="true">—</span><label htmlFor="end-date">End date<input id="end-date" name="end" type="date" value={end} aria-invalid={!!error} aria-describedby={error ? 'date-error' : 'date-help'} onInput={event => { setEnd(event.currentTarget.value); setError(''); }}/></label></div>
      <div className="date-actions"><button className="primary-button" type="submit">Apply period</button><button className="date-reset" type="button" onClick={() => { setStart(''); setEnd(''); setError(''); save(null); }}><RotateCcw size={13}/> Reset</button></div>
    </form>
    <div className="date-filter-status"><span aria-live="polite"><strong>{formatRange(range)}</strong>{pending && <span className="pending-date"> · Changes not applied</span>}</span><span id="date-help">Vancouver calendar dates · Apply to update reports and rainfall.</span></div>
    {error && <p id="date-error" className="date-error" role="alert">{error}</p>}
    {storageWarning && <p className="date-error" role="status">{storageWarning}</p>}
  </section>;
}




