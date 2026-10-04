import { Download, MapPin, X } from 'lucide-react';
import type { SelectedCell } from './CityMap';

export const checks = ['Grate checked', 'Debris checked', 'Standing water checked', 'Damage checked'] as const;
export const statuses = ['Planned', 'Inspected', 'Blockage found', 'Cleared'] as const;
export type Inspection = SelectedCell & { status: typeof statuses[number]; checks: boolean[]; notes: string; probability: number | null; windowStart: string | null; windowEnd: string | null };
export const storageKey = 'vanpredict-inspections-v1';
export function readPlan(): Inspection[] {
  try {
    const data: unknown = JSON.parse(localStorage.getItem(storageKey) ?? '[]');
    if (!Array.isArray(data)) return [];
    return data.filter((item): item is Inspection => !!item && typeof item === 'object' && typeof item.id === 'string' && item.id.startsWith('g500_') && item.size === 500 && Number.isFinite(item.area) && statuses.includes(item.status) && Array.isArray(item.checks) && item.checks.length === 4 && item.checks.every((v: unknown) => typeof v === 'boolean') && typeof item.notes === 'string' && (item.probability === null || (Number.isFinite(item.probability) && item.probability >= 0 && item.probability <= 1)) && (item.windowStart === null || typeof item.windowStart === 'string') && (item.windowEnd === null || typeof item.windowEnd === 'string'));
  } catch { return []; }
}
export function exportPlan(plan: Inspection[]) {
  // Prefix spreadsheet formula characters in user-supplied notes/IDs.
  const quote = (value: unknown) => { let text = value == null ? '' : String(value); if (/^[\s]*[=+@-]/.test(text)) text = "'" + text; return '"' + text.replaceAll('"', '""') + '"'; };
  const rows: unknown[][] = [['Area ID','Status','Report probability (%)','Model window start','Model window end',...checks,'Notes','Estimate meaning']];
  for (const item of plan) rows.push([item.id,item.status,item.probability === null ? '' : (item.probability * 100).toFixed(2),item.windowStart,item.windowEnd,...item.checks.map(value => value ? 'Yes' : 'No'),item.notes,'Experimental drainage-report estimate; not flood probability']);
  return '\uFEFF' + rows.map(row => row.map(quote).join(',')).join('\r\n');
}
export default function InspectionPlan({plan,onChange,onSelect,saveError}:{plan:Inspection[];onChange:(plan:Inspection[])=>void;onSelect:(cell:SelectedCell)=>void;saveError:boolean}) {
  function update(id:string,patch:Partial<Inspection>) { onChange(plan.map(item => item.id === id ? {...item,...patch} : item)); }
  function download() {
    const url = URL.createObjectURL(new Blob([exportPlan(plan)],{type:'text/csv;charset=utf-8'}));
    const link=document.createElement('a'); link.href=url;link.download=`vanpredict-inspections-${new Date().toISOString().slice(0,10)}.csv`;document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  return <section id="inspection-plan" className="inspection-plan" aria-labelledby="inspection-plan-title"><div className="plan-heading"><div><span className="eyebrow">FROM INSIGHT TO ACTION</span><h2 id="inspection-plan-title">Your inspection shortlist</h2><p>Saved in this browser. Record field checks and export your plan.</p></div><button className="primary-button" disabled={!plan.length} onClick={download}><Download size={15}/> Export CSV</button></div>
    {saveError && <p role="alert">Browser storage is unavailable. Export your plan before leaving this page.</p>}
    {!plan.length ? <p className="plan-empty">Select a grid cell on the map and add it to your plan.</p> : <div className="inspection-records">{plan.map((item,index)=><article className="inspection-record" key={item.id}><div className="record-heading"><span className="plan-number">{String(index+1).padStart(2,'0')}</span><button onClick={()=>onSelect(item)}><MapPin size={15}/>{item.id}</button><button className="plan-remove" aria-label={`Remove ${item.id} from inspection plan`} onClick={()=>onChange(plan.filter(other=>other.id!==item.id))}><X size={16}/></button></div><div className="record-body"><label>Status<select aria-label={`Inspection status for ${item.id}`} value={item.status} onChange={event=>update(item.id,{status:event.target.value as Inspection['status']})}>{statuses.map(status=><option key={status}>{status}</option>)}</select></label><span className="record-estimate">{item.probability===null?'No model estimate saved':`${(item.probability*100).toFixed(1)}% report probability at time added`}</span><fieldset><legend>Field checklist</legend>{checks.map((check,i)=><label key={check}><input type="checkbox" checked={item.checks[i]} onChange={event=>update(item.id,{checks:item.checks.map((value,j)=>j===i?event.target.checked:value)})}/>{check}</label>)}</fieldset><label className="record-notes">Inspection notes<textarea maxLength={2000} aria-label={`Inspection notes for ${item.id}`} value={item.notes} placeholder="Access, observations, or follow-up needed…" onChange={event=>update(item.id,{notes:event.target.value})}/></label></div></article>)}</div>}
    <p className="plan-note">Checklist ticks record checks performed, not a finding that an asset is clear. Status changes are manual. Records stay on this device; no crews are dispatched.</p></section>;
}
