import { useEffect, useRef, useState } from 'react';
import type { SelectedCell } from './CityMap';
export type ModelResult = {prediction_start:string;prediction_end:string;generated_at:string;input_age_hours:number;model_version:string;history_cutoff:string;areas:{cell_id:string;report_probability:number;rank:number}[];top_areas:{cell_id:string;report_probability:number;rank:number}[]};
const time=(value:string)=>new Intl.DateTimeFormat('en-CA',{month:'short',day:'numeric',year:'numeric',hour:'numeric',minute:'2-digit',timeZone:'America/Vancouver'}).format(new Date(value));
export default function HistoricalPrediction({gridSize,result,onResult,onSelect}:{gridSize:number;result:ModelResult|null;onResult:(result:ModelResult|null)=>void;onSelect:(cell:SelectedCell)=>void}){
  const [date,setDate]=useState('2025-06-04');
  const dateInput=useRef<HTMLInputElement>(null);
  const [loading,setLoading]=useState(false);
  const [error,setError]=useState('');
  const controller=useRef<AbortController|null>(null);
  useEffect(()=>{controller.current?.abort();setLoading(false);setError('');onResult(null);return()=>controller.current?.abort();},[gridSize,onResult]);
  async function run(){
    const date=dateInput.current?.value; if(!date){setError("Choose a prediction date.");return;}
    controller.current?.abort();const request=new AbortController();controller.current=request;
    setLoading(true);setError('');onResult(null);
    try{
      const offset=new Intl.DateTimeFormat('en',{timeZone:'America/Vancouver',timeZoneName:'longOffset'}).formatToParts(new Date(`${date}T12:00:00Z`)).find(p=>p.type==='timeZoneName')!.value.replace('GMT','');
      const response=await fetch('/api/v1/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prediction_time:`${date}T06:00:00${offset}`,grid_size_m:500,forecast_hours:36,top_k:5}),signal:AbortSignal.any([request.signal,AbortSignal.timeout(30000)])});
      const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'Prediction could not load. Check the date and try again.');
      if(!Array.isArray(data.areas)||data.areas.length!==604||data.areas.some((a:{report_probability:number})=>!Number.isFinite(a.report_probability)||a.report_probability<0||a.report_probability>1))throw new Error('Unexpected model response.');
      if(!request.signal.aborted)onResult(data);
    }catch(failure){if(!request.signal.aborted)setError(failure instanceof Error?failure.message:'Prediction unavailable.');}
    finally{if(!request.signal.aborted)setLoading(false);}
  }
  async function select(id:string){
    try{
      const response=await fetch('/grids/vancouver_500m.geojson');if(!response.ok)throw new Error('Grid unavailable');
      const grid=await response.json();const cell=grid.features.find((f:{properties:{grid_id:string}})=>f.properties.grid_id===id)?.properties;
      if(cell)onSelect({id,size:500,area:cell.area_m2});
    }catch{setError('The selected area could not load. Please try again.');}
  }
  return <section className="prediction-panel" aria-labelledby="history-model-title" aria-busy={loading}>
    <div className="prediction-intro"><div><div className="eyebrow">MODEL EXPLORER</div><h2 id="history-model-title">Where reports may occur.</h2><p className="prediction-message">Experimental 36-hour predictions using historical inputs.</p></div><span className="demo-badge">EXPERIMENTAL · HISTORICAL</span></div>
    <div className="model-controls"><label>Prediction date<input type="date" ref={dateInput} defaultValue="2025-06-04" onChange={e=>{controller.current?.abort();setLoading(false);setError('');onResult(null);setDate(e.target.value);}}/></label><button className="primary-button" disabled={loading||gridSize!==500||!date} onClick={()=>void run()}>{loading?'Scoring areas…':'Predict historical 36 hours'}</button>{result&&<button onClick={()=>onResult(null)}>Clear predictions</button>}</div>
    <p className="prediction-disclaimer">Starts at 6 a.m. Vancouver time. This model uses past rain and reports; the live forecast above does not influence these scores. Historical period filters are separate.{gridSize!==500&&' Switch to the 500 m grid to predict.'}</p>
    <div role="status">{loading&&<p>Preparing inputs and scoring all 604 cells…</p>}</div>
    {error&&<p className="prediction-error" role="alert">{error} <button onClick={()=>void run()} disabled={gridSize!==500}>Retry prediction</button></p>}
    {result&&<div className="prediction-results"><div className="forecast-summary"><span>Historical prediction window</span><strong>{time(result.prediction_start)} → {time(result.prediction_end)}</strong><small>Input age: {(result.input_age_hours/24).toFixed(1)} days · {result.model_version}</small><small>Report history available through {time(result.history_cutoff)}</small><p className="prediction-message">Estimated chance of a matching drainage report, not flooding. Probabilities remain experimental.</p></div><div className="prediction-ranked"><h3>Top 5 areas to explore</h3><ol>{result.top_areas.map(area=><li key={area.cell_id}><button onClick={()=>void select(area.cell_id)} aria-label={`Explore ranked area ${area.rank}: ${area.cell_id}`}><span>{area.rank}</span><span className="prediction-area">{area.cell_id}<small>Matching report estimate</small></span><strong>{(area.report_probability*100).toFixed(1)}%</strong></button></li>)}</ol></div></div>}
  </section>;
}
