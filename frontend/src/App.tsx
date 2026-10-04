import { useCallback, useEffect, useState } from 'react';
import { ArrowDownLeft, CloudRain, Droplets, Info, Layers3, MapPin, Radio, ShieldCheck, X } from 'lucide-react';
import CityMap, { type SelectedCell } from './CityMap';
import HistoricalPrediction, { type ModelResult } from './HistoricalPrediction';
import PredictionPanel from './PredictionPanel';
import usePredictionPreview from './usePredictionPreview';
import InspectionPlan, { readPlan, storageKey, type Inspection } from './InspectionPlan';

export default function App() {
  const gridSize = 500;
  const [cellCount, setCellCount] = useState<number | null>(null);
  const [modelResult, setModelResult] = useState<ModelResult | null>(null);
  const [selectedCell, setSelectedCell] = useState<SelectedCell | null>(null);
  const [inspectionList, setInspectionList] = useState<Inspection[]>(readPlan);
  const [saveError, setSaveError] = useState(false);
  const [showBasins, setShowBasins] = useState(true);
  useEffect(() => { try { localStorage.setItem(storageKey, JSON.stringify(inspectionList)); setSaveError(false); } catch { setSaveError(true); } }, [inspectionList]);
  const prediction = usePredictionPreview(gridSize);
  const receiveModel = useCallback((result: ModelResult | null) => setModelResult(result), []);
  const scores = modelResult ? Object.fromEntries(modelResult.top_areas.map(area => [area.cell_id, area.report_probability])) : null;
  const selectedScore = modelResult?.areas.find(area => area.cell_id === selectedCell?.id)?.report_probability;
  const metrics = [
    { label: 'Forecast rainfall', value: prediction.data?.totalRainMm.toFixed(2) ?? '—', note: 'mm · next 36 hours', foot: prediction.loading ? 'Fetching weather forecast' : prediction.error ? 'Forecast unavailable · retry below' : prediction.data ? 'Live forecast · central Vancouver' : 'Load the forecast below', icon: <CloudRain size={19}/>, tone: 'blue' },
    { label: 'Prediction window', value: '36', note: 'hours per model run', foot: 'Experimental drainage-report estimates', icon: <Droplets size={19}/>, tone: 'teal' },
    { label: 'Areas assessed', value: cellCount ?? '—', note: '500 m Vancouver grids', foot: modelResult ? 'Top 5 areas highlighted on the map' : 'Run the model to compare areas', icon: <Layers3 size={19}/>, tone: 'sand' },
  ];
  return <>
    <a className="skip-link" href="#main">Skip to predictions</a>
    <header className="topbar">
      <a href="#main" className="brand" aria-label="VanPredict dashboard"><span className="brand-mark"><Droplets size={23}/></span><span>van<span className="brand-light">predict</span><span className="brand-dot">.</span></span></a>
      <nav aria-label="Main navigation"><a href="#prediction-tools">Prepare</a><a href="#city-explorer">Prioritize</a><a href="#inspection-plan">Inspection plan <span className="nav-count">{inspectionList.length}</span></a></nav>
      <div className="city-label"><MapPin size={14}/> Vancouver, BC <span className="city-dot"/></div>
    </header>
    <main id="main" className="page">
      <section className="intro" aria-labelledby="page-title">
        <div className="hero-copy"><div className="eyebrow"><span/> PREVENTIVE DRAINAGE INSPECTIONS</div><h1 id="page-title">Before the storm.<br/><span>Know where to inspect.</span></h1><p>Help drainage teams focus preventive inspections where they matter most.</p><div className="hero-links"><a href="#prediction-tools">Build an inspection plan <ArrowDownLeft size={16}/></a><span><MapPin size={13}/> Vancouver, British Columbia</span></div></div>
        <div className="hero-art" aria-hidden="true"><svg viewBox="0 0 380 240" fill="none"><path d="M-40 178C35 62 98 260 174 119S305 15 410 89M-40 194C36 78 98 276 174 135S305 31 410 105M-40 210C36 94 98 292 174 151S305 47 410 121M-40 226C36 110 98 308 174 167S305 63 410 137M-40 162C35 46 98 244 174 103S305-1 410 73M-40 146C35 30 98 228 174 87S305-17 410 57M-40 130C35 14 98 212 174 71S305-33 410 41"/><circle cx="237" cy="110" r="41"/><circle cx="237" cy="110" r="65"/><circle cx="237" cy="110" r="4" className="hero-pin"/></svg><div className="hero-coordinate">49.2827° N &nbsp; 123.1207° W</div></div>
        <span className="preview-pill"><Radio size={14}/> PRE-STORM PLANNING</span>
      </section>
      <section className="purpose-strip" aria-label="VanPredict use case"><div><span className="eyebrow">THE VANPREDICT VISION</span><p>Historical municipal data <b>+</b> infrastructure characteristics <b>+</b> live weather forecasts <span>→</span> <strong>smarter inspection priorities before a storm.</strong></p></div><details><summary>What is connected today?</summary><p>The current model uses historical rainfall and municipal reports. Live weather is displayed separately; infrastructure features remain in research models.</p></details></section>
      <section className="metrics" aria-label="Prediction overview">{metrics.map(metric => <article className="metric" key={metric.label}><div className="metric-heading"><span>{metric.label}</span><span className={`metric-icon ${metric.tone}`}>{metric.icon}</span></div><div className="metric-value">{metric.value}<span>{metric.note}</span></div><div className="metric-foot"><span className="tiny-dot"/>{metric.foot}</div></article>)}</section>
      <div className="workflow-steps" aria-label="Inspection planning workflow"><a href="#prediction-tools"><span>01</span><div><strong>Check the weather</strong><small>See the next 36 hours</small></div></a><a href="#prediction-tools"><span>02</span><div><strong>Compare priorities</strong><small>Run the model and review estimates</small></div></a><a href="#city-explorer"><span>03</span><div><strong>Plan inspections</strong><small>Select areas on one shared map</small></div></a></div>
      <div id="prediction-tools" className="forecast-tools"><PredictionPanel preview={prediction} onRun={() => { void prediction.run(); }} onClear={prediction.clear}/><HistoricalPrediction gridSize={gridSize} result={modelResult} onResult={receiveModel} onSelect={setSelectedCell}/></div>
      <div className="section-label"><span>YOUR INSPECTION WORKSPACE</span><span>One map. Ranked areas. A focused plan.</span></div>
      <section id="city-explorer" className="workspace" aria-label="Prediction map">
        <div className="map-card">
          <div className="map-heading"><div><h2>Inspection priority map</h2><p>{modelResult ? 'Top 5 areas by estimated drainage-report probability' : 'Run the model to highlight its highest-ranked areas'}</p></div><span className="small-tag">{modelResult ? 'MODEL RESULTS' : 'AWAITING PREDICTION'}</span></div>
          <div className="map-toolbar"><span className="layer-label"><Layers3 size={15}/> {modelResult ? 'Model estimates' : 'Vancouver coverage'}</span><label className="basin-toggle"><input type="checkbox" checked={showBasins} onChange={event => setShowBasins(event.target.checked)}/> Catch basins in selected area</label></div>
          <CityMap gridSize={gridSize} onLoad={setCellCount} selected={selectedCell} onSelect={setSelectedCell} counts={null} reportLayer={false} predictionScores={scores} showBasins={showBasins}/>
          <div className="map-footer"><span>{modelResult ? 'Darker = higher estimate among highlighted areas' : 'No scores shown until the model runs'}</span><span><Info size={13}/> Experimental report estimates</span></div>
        </div>
        <aside className="details-card" aria-labelledby="details-title"><div className="details-heading"><h2 id="details-title">Area prediction</h2><span className="small-tag">{selectedCell ? 'SELECTED' : 'EXPLORE'}</span></div>
          {selectedCell ? <div className="selected-insights" aria-live="polite"><span className="selected-icon"><MapPin size={27}/></span><h3>Your selected area</h3><p className="cell-id">{selectedCell.id}</p><div className="area-probability"><strong>{selectedScore === undefined ? '—' : `${(selectedScore * 100).toFixed(1)}%`}</strong><span>Estimated drainage-report probability</span></div><button className="primary-button shortlist-button" disabled={inspectionList.some(cell => cell.id === selectedCell.id)} onClick={() => setInspectionList(list => [...list, {...selectedCell, status: 'Planned', checks: [false,false,false,false], notes: '', probability: selectedScore ?? null, windowStart: modelResult?.prediction_start ?? null, windowEnd: modelResult?.prediction_end ?? null}])}>{inspectionList.some(cell => cell.id === selectedCell.id) ? 'Added to inspection plan' : 'Add to inspection plan'}</button><button className="clear-selection" onClick={() => setSelectedCell(null)}><X size={14}/> Clear selection</button></div> : <div className="empty-insights"><div className="location-art" aria-hidden="true"><div className="location-ring"/><MapPin size={34} strokeWidth={1.4}/><span className="orbit-dot"/></div><h3>A closer look at risk.</h3><p>Choose a highlighted area to see its model estimate.</p><span className="selection-hint"><ArrowDownLeft size={15}/> Select a place on the map</span></div>}
          <div className="insight-rows"><div><span>Forecast horizon</span><strong>36 hours</strong></div><div><span>Model inputs</span><strong>Historical rainfall & reports</strong></div>{modelResult && <div className="period-insight"><span>Model window · Vancouver time</span><strong>{new Date(modelResult.prediction_start).toLocaleString('en-CA', { timeZone: 'America/Vancouver' })} → {new Date(modelResult.prediction_end).toLocaleString('en-CA', { timeZone: 'America/Vancouver' })}</strong></div>}</div>
          <div className="model-note"><ShieldCheck size={18}/><div><strong>Understanding the estimate</strong><p>Scores estimate drainage reports, not confirmed flooding. Live weather is displayed separately from current model inputs.</p></div></div>
        </aside>
      </section>
      <InspectionPlan plan={inspectionList} onChange={setInspectionList} saveError={saveError} onSelect={cell => { setSelectedCell(cell); document.getElementById('city-explorer')?.scrollIntoView({ behavior: 'auto' }); }}/>
      <footer className="page-footer"><span>VANPREDICT <span className="footer-divider">/</span> 36-hour drainage prediction explorer</span><span>Experimental estimates · not flood warnings.</span></footer>
    </main>
  </>;
}


