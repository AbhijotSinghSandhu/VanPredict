import { useEffect, useRef, useState } from 'react';
import { requestLiveForecast } from './liveForecast';
import type { Forecast } from './liveForecast';

export default function usePredictionPreview(size: number) {
  const controller = useRef<AbortController | null>(null);
  const [result,setResult] = useState<Forecast | null>(null);
  const [loading,setLoading] = useState(false);
  const [error,setError] = useState('');
  function clear() { controller.current?.abort(); setResult(null); setError(''); setLoading(false); }
  useEffect(()=>{ clear(); return ()=>controller.current?.abort(); },[size]);
  async function run() {
    controller.current?.abort();
    const request = new AbortController(); controller.current=request;
    setLoading(true); setResult(null); setError('');
    try {
      const data = await requestLiveForecast(size,request.signal);
      if (request.signal.aborted) return null;
      setResult(data); return data;
    } catch (failure) {
      if (!request.signal.aborted) setError(failure instanceof Error ? failure.message : 'Preview unavailable. Try again.');
      return null;
    } finally { if (!request.signal.aborted) setLoading(false); }
  }
  return { data: result?.gridSize === size ? result : null, loading, error, run, clear };
}
