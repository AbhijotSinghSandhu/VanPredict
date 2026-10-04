import { useEffect, useState } from 'react';
import type { DateRange } from './DateFilters';

export type ReportSummary = {
  start: string; end: string; coverage_start: string; coverage_end: string;
  total_reports: number; mapped_reports: number; unmapped_reports: number;
  partial_coverage: boolean; grid_size: number; counts: Record<string, number>; source: string;
};

export default function useReports(range: DateRange | null, size: number) {
  const [data, setData] = useState<ReportSummary | null>(null);
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setData(null); setStatus('loading');
    const query = new URLSearchParams({ grid_size: String(size) });
    if (range) { query.set('start', range.start); query.set('end', range.end); }
    fetch(`/api/v1/reports?${query}`, { signal: controller.signal })
      .then(response => { if (!response.ok) throw new Error('Reports unavailable'); return response.json() as Promise<ReportSummary>; })
      .then(result => { if (!controller.signal.aborted) { setData(result); setStatus('ready'); } })
      .catch(() => { if (!controller.signal.aborted) setStatus('error'); });
    return () => controller.abort();
  }, [range, size, attempt]);
  // Never show an old grid or period while a new request starts.
  const current = data?.grid_size === size && (range ? (data.start === range.start && data.end === range.end) : (data.start === data.coverage_start && data.end === data.coverage_end)) ? data : null;
  return { data: current, status: status === 'ready' && !current ? 'loading' : status, retry: () => setAttempt(value => value + 1) };
}
