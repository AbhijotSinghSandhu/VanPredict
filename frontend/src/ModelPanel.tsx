import { useEffect, useState } from 'react';
import { FlaskConical } from 'lucide-react';

type Metrics = { precision_top20_per_day: number; recall_top20_per_day: number; average_precision: number; roc_auc: number; rows: number; positive_grid_days: number; prevalence: number };
type Evaluation = { version: string; test: Metrics; location_baseline_test: Metrics; splits_by_target_date: { training: string[]; calibration: string[]; test: string[] }; artifact_available: boolean; limitations: string[] };
const percent = (number: number) => `${(number*100).toFixed(2)}%`;

export default function ModelPanel() {
  const [data,setData] = useState<Evaluation | null>(null);
  const [error,setError] = useState(false);
  const [attempt,setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController(); setError(false);
    fetch('/api/v1/model-status',{signal:controller.signal}).then(response=>{if(!response.ok)throw new Error('Unavailable');return response.json() as Promise<Evaluation>;})
      .then(result=>{if(!controller.signal.aborted)setData(result);}).catch(()=>{if(!controller.signal.aborted)setError(true);});
    return ()=>controller.abort();
  },[attempt]);
  return <section className="model-panel" aria-labelledby="model-title"><div className="rainfall-heading"><div><h2 id="model-title"><FlaskConical size={18}/> Earlier next-day research</h2><p>Testing a 500 m model for next-day drainage reports</p></div><span className="small-tag">EXPERIMENTAL · LIVE PREDICTIONS UNAVAILABLE</span></div>
    {!data && <p className="rainfall-message" role="status">{error ? <><span>Evaluation could not load.</span> <button onClick={()=>setAttempt(value=>value+1)}>Retry evaluation</button></> : 'Loading the model evaluation…'}</p>}
    {data && <><p className="model-finding">The model improves overall ranking, but its daily shortlist does not yet beat the location-only baseline.</p><div className="model-comparison"><div><span>Model shortlist precision</span><strong>{percent(data.test.precision_top20_per_day)}</strong><p>Flagged cell-days with a matching report</p></div><div><span>Location-only baseline</span><strong>{percent(data.location_baseline_test.precision_top20_per_day)}</strong><p>Using each area's past report frequency</p></div><div><span>Reported cell-days found</span><strong>{percent(data.test.recall_top20_per_day)}</strong><p>Baseline: {percent(data.location_baseline_test.recall_top20_per_day)}</p></div></div><p className="rainfall-source">Shortlists select 20 of 604 cells each day. Results use held-out target dates {data.splits_by_target_date.test[0]} to {data.splits_by_target_date.test[1]}; these figures stay fixed when you change dashboard dates. Most flagged cell-days had no matching report.</p>
    <details className="rainfall-table"><summary>Evaluation details and limitations</summary><div className="model-details"><p>Training target dates: {data.splits_by_target_date.training.join(' to ')}. Calibration target dates: {data.splits_by_target_date.calibration.join(' to ')}.</p><p>Test set: {data.test.rows.toLocaleString()} grid-days; {data.test.positive_grid_days.toLocaleString()} with matching reports ({percent(data.test.prevalence)}).</p><p>Average precision: {data.test.average_precision.toFixed(4)}; baseline {data.location_baseline_test.average_precision.toFixed(4)}. ROC-AUC: {data.test.roc_auc.toFixed(4)}.</p><ul>{data.limitations.map(note=><li key={note}>{note}</li>)}</ul><p>Version: {data.version} · Saved model {data.artifact_available ? 'available' : 'missing'}</p></div></details></>}
  </section>;
}
