import { useEffect, useState } from 'react';
import type { DateRange } from './DateFilters';
type Stats = { total_mm: number | null; daily_average_mm: number | null; covered_days: number; requested_days: number; complete: boolean };
export type RainfallSummary = { start: string; end: string; coverage_start: string; coverage_end: string; city: Stats; cell: Stats | null; cell_id: string | null; grid_size: number; weather_cell_id: string | null; daily: { date: string; mm: number | null }[] };

export default function useRainfall(range: DateRange | null, size: number, cell: string | null) {
  const key = `${range?.start ?? ''}/${range?.end ?? ''}/${size}/${cell ?? ''}`;
  const [result, setResult] = useState<{ key: string; data: RainfallSummary } | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController(); setResult(null); setFailed(null);
    const query = new URLSearchParams({ grid_size: String(size) });
    if (range) { query.set('start', range.start); query.set('end', range.end); }
    if (cell) query.set('cell_id', cell);
    fetch(`/api/v1/rainfall?${query}`, { signal: controller.signal })
      .then(response => { if (!response.ok) throw new Error('Unavailable'); return response.json() as Promise<RainfallSummary>; })
      .then(data => { if (!controller.signal.aborted) setResult({ key, data }); })
      .catch(() => { if (!controller.signal.aborted) setFailed(key); });
    return () => controller.abort();
  }, [key, attempt]);
  return { data: result?.key === key ? result.data : null, error: failed === key, retry: () => setAttempt(value => value + 1) };
}
