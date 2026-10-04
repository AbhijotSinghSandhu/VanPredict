import { useRef, useState } from 'react';
import CityMap from './CityMap';
import type { SelectedCell } from './CityMap';
import { reportBins } from './reportColours';
import useRainfall from './useRainfall';
import RainfallPanel from './RainfallPanel';
import ModelPanel from './ModelPanel';
import PredictionPanel from './PredictionPanel';
import usePredictionPreview from './usePredictionPreview';
import useReports from './useReports';
import DateFilters, { readSavedRange, formatRange } from './DateFilters';
import { ArrowDownLeft, ArrowUpRight, CloudRain, Droplets, Info, Layers3, MapPin, MoveUpRight, Radio, ShieldCheck, X } from 'lucide-react';

export default function App() {
  const [dateRange, setDateRange] = useState(readSavedRange);
  const [cellCount, setCellCount] = useState<number | null>(null);
  const [gridSize, setGridSize] = useState(500);
  const [reportLayer, setReportLayer] = useState(true);
  const prediction = usePredictionPreview(gridSize);
  const [selectedCell, setSelectedCell] = useState<SelectedCell | null>(null);
  const reports = useReports(dateRange, gridSize);
  const rainfall = useRainfall(dateRange, gridSize, selectedCell?.id ?? null);
  const aboutDialog = useRef<HTMLDialogElement>(null);
  return <>
    <a className="skip-link" href="#main">Skip to dashboard</a>
    <header className="topbar">
      <a href="#main" className="brand" aria-label="VanPredict dashboard"><span className="brand-mark"><Droplets size={23}/></span><span>van<span className="brand-light">predict</span><span className="brand-dot">.</span></span></a>
      <nav aria-label="Main navigation"><a className="nav-active" href="#main">Overview</a><button onClick={() => aboutDialog.current?.showModal()}>About the project <ArrowUpRight size={14}/></button></nav>
      <div className="city-label"><MapPin size={14}/> Vancouver, BC <span className="city-dot"/></div>
    </header>
    <main id="main" className="page">
      <section className="intro" aria-labelledby="page-title">
        <div><div className="eyebrow"><span/> A CLOSER LOOK AT OUR CITY</div><h1 id="page-title">Understand the rain.<br className="mobile-break"/> See the bigger picture.</h1><p>Exploring drainage, rainfall, and the places where they meet.</p></div>
        <span className="preview-pill"><Radio size={14}/> Interface preview</span>
      </section>
      <section className="metrics" aria-label="Dashboard metrics preview">
        {[
          { label: 'Drainage reports', icon: <Droplets size={19}/>, note: 'Historical 311 requests', foot: reports.status === 'loading' ? 'Loading historical reports' : reports.status === 'error' ? 'Reports unavailable' : 'Prepared Vancouver 311 data', tone: 'teal' },
          { label: 'Rainfall', icon: <CloudRain size={19}/>, note: 'mm/day · city daily average', foot: rainfall.error ? 'Rainfall unavailable' : !rainfall.data ? 'Loading rainfall' : rainfall.data.city.complete ? 'ERA5 estimate · average over covered days' : 'Available days · incomplete coverage', tone: 'blue' },
          { label: 'Areas to explore', icon: <Layers3 size={19}/>, note: 'Vancouver grid cells', foot: cellCount === null ? 'Loading grid geometry' : 'Real Vancouver grid loaded', tone: 'sand' },
        ].map(metric => <article className="metric" key={metric.label}><div className="metric-heading"><span>{metric.label}</span><span className={`metric-icon ${metric.tone}`}>{metric.icon}</span></div><div className="metric-value">{metric.label === 'Areas to explore' ? (cellCount ?? '\u2014') : metric.label === 'Drainage reports' ? (reports.data?.total_reports.toLocaleString() ?? '\u2014') : metric.label === 'Rainfall' ? (rainfall.data?.city.daily_average_mm?.toFixed(2) ?? '\u2014') : '\u2014'}<span>{metric.note}</span></div><div className="metric-foot"><span className="tiny-dot"/>{metric.foot}</div></article>)}
      </section>
      <DateFilters range={dateRange} onApply={setDateRange}/>
      <div className="reports-status" role="status">
        {reports.status === 'loading' && 'Loading historical drainage reports…'}
        {reports.status === 'error' && <><span>Drainage reports could not load. Check that the backend is running.</span><button onClick={reports.retry}>Try again</button></>}
        {reports.data && <><span>Report dates: {reports.data.coverage_start} to {reports.data.coverage_end} · {dateRange ? 'Selected period' : 'All available dates'}{reports.data.total_reports === 0 ? ' · No reports in this period' : ''}</span>{reports.data.partial_coverage && <strong>Selected period extends beyond the report date span.</strong>}{reports.data.unmapped_reports > 0 && <strong>{reports.data.unmapped_reports} reports could not be assigned to this grid.</strong>}</>}
      </div>
      <PredictionPanel preview={prediction} onRun={()=>{void prediction.run();}} onClear={prediction.clear}/>
      <section className="workspace" aria-label="City explorer">
        <div className="map-card">
          <div className="map-heading"><div><h2>City explorer</h2><p>A neighbourhood-level perspective</p></div><span className="small-tag">HISTORICAL VIEW</span></div>
          <div className="map-toolbar"><label className="grid-picker layer-picker"><Layers3 size={15}/><select aria-label="Map layer" value={reportLayer ? 'reports' : 'grid'} onChange={event => {setReportLayer(event.target.value === 'reports');}}><option value="reports">Drainage reports</option><option value="grid">Grid coverage</option></select></label><label className="grid-picker">Cell size <select aria-label="Grid cell size" value={gridSize} onChange={event => { const nextSize=Number(event.target.value); if(nextSize===gridSize)return; prediction.clear();setCellCount(null); setSelectedCell(null); setGridSize(nextSize); }}><option value={200}>200 m</option><option value={500}>500 m</option><option value={1000}>1,000 m</option></select></label></div>
          <CityMap gridSize={gridSize} onLoad={setCellCount} selected={selectedCell} onSelect={setSelectedCell} counts={reports.data?.counts ?? null} reportLayer={reportLayer} predictionScores={null}/>
          <div className="map-footer report-map-footer">{reportLayer ? <div className="report-legend" aria-label="Reports per cell legend"><span>Reports / cell</span>{reportBins.map(bin => <span key={bin.label}><i style={{ background: bin.colour }}/>{bin.label}</span>)}</div> : <span><span className="legend-square"/> Grid coverage</span>}<span><Info size={13}/>Historical counts · not predicted risk</span></div>
        </div>
        <aside className="details-card" aria-labelledby="details-title"><div className="details-heading"><h2 id="details-title">Area insights</h2><span className="small-tag">{selectedCell ? 'SELECTED' : 'EXPLORE'}</span></div>
          {selectedCell ? <div className="selected-insights" aria-live="polite"><span className="selected-icon"><MapPin size={27}/></span><h3>Your selected area</h3><p className="cell-id">{selectedCell.id}</p><div className="cell-facts"><div><span>Grid size</span><strong>{selectedCell.size.toLocaleString()} m</strong></div><div><span>Area within city</span><strong>{Math.round(selectedCell.area).toLocaleString()} m&sup2;</strong></div></div><p className="cell-explanation">Cells at the city boundary can be smaller than a full square.</p><button className="clear-selection" onClick={() => setSelectedCell(null)}><X size={14}/> Clear selection</button></div> : <div className="empty-insights"><div className="location-art" aria-hidden="true"><div className="location-ring"/><MapPin size={34} strokeWidth={1.4}/><span className="orbit-dot"/></div><h3>Every area has a story.</h3><p>Select a grid cell to explore its location and dimensions.</p><span className="selection-hint"><ArrowDownLeft size={15}/> Choose a place on the map</span></div>}
          <div className="insight-rows"><div className="period-insight"><span>Selected period</span><strong>{formatRange(dateRange ?? (reports.data ? { start: reports.data.start, end: reports.data.end } : null))}</strong></div><div><span>Reports in this area</span><strong>{reports.status === 'loading' ? 'Loading…' : reports.status === 'error' ? 'Unavailable' : selectedCell && reports.data ? (reports.data.counts[selectedCell.id] ?? 0).toLocaleString() : 'Select an area'}</strong></div><div><span>Area daily average rainfall</span><strong>{rainfall.error ? 'Unavailable' : !selectedCell ? 'Select an area' : !rainfall.data ? 'Loading…' : rainfall.data.cell?.daily_average_mm == null ? 'No data' : `${rainfall.data.cell.daily_average_mm.toFixed(2)} mm/day`}</strong></div><div><span>Area rainfall estimate</span><strong>{rainfall.error ? 'Unavailable' : !selectedCell ? 'Select an area' : !rainfall.data ? 'Loading…' : rainfall.data.cell?.total_mm == null ? 'No data' : `${rainfall.data.cell.total_mm.toLocaleString()} mm`}</strong></div>{selectedCell && rainfall.data?.cell && <div><span>Rainfall days covered</span><strong>{rainfall.data.cell.covered_days} / {rainfall.data.cell.requested_days}</strong></div>}</div><div className="model-note"><ShieldCheck size={18}/><div><strong>Thoughtful about uncertainty</strong><p>Predictions will appear once the model has been validated.</p></div></div></aside>
      </section>
      <RainfallPanel data={rainfall.data} error={rainfall.error} retry={rainfall.retry}/>
      <ModelPanel/>
      <section className="context-strip"><span className="context-icon"><MoveUpRight size={19}/></span><div><h3>Built for a better understanding of Vancouver.</h3><p>Connecting public drainage reports with rainfall and spatial data, one layer at a time.</p></div><button onClick={() => aboutDialog.current?.showModal()}>Our approach <ArrowUpRight size={16}/></button></section>
      <footer className="page-footer"><span>VANPREDICT <span className="footer-divider">/</span> Vancouver drainage explorer</span><span>Historical reports are not confirmed flood events.</span></footer>
    </main>
    <dialog ref={aboutDialog} className="about-dialog" aria-labelledby="about-title"><div className="dialog-heading"><span className="eyebrow">THE PROJECT</span><button className="close-button" aria-label="Close about dialog" onClick={() => aboutDialog.current?.close()}><X size={20}/></button></div><h2 id="about-title">A clearer view of drainage.</h2><p>VanPredict brings together Vancouver's historical 311 drainage reports, rainfall records, and spatial data to explore patterns across the city.</p><p>This prototype displays the real Vancouver grid over a street map. Drainage counts show prepared historical 311 reports for the selected period. Rainfall is an ERA5 liquid-rain estimate; two weather source cells provide the weather detail. Select a cell to inspect its dimensions, or switch grid sizes to explore at a different scale.</p><div className="dialog-note"><ShieldCheck size={21}/><p>A drainage report does not confirm flooding. Model estimates will be clearly labelled and only added after validation.</p></div><button className="primary-button" onClick={() => aboutDialog.current?.close()}>Back to the explorer <ArrowUpRight size={16}/></button></dialog>
  </>;
}






