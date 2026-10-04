import { useEffect, useRef, useState } from 'react';
import * as L from 'leaflet';
import type { FeatureCollection } from 'geojson';
import 'leaflet/dist/leaflet.css';
import { reportColour } from './reportColours';

const gridCache = new Map<number, FeatureCollection>();
export type SelectedCell = { id: string; size: number; area: number };
const cellStyle = { color: '#327c69', weight: 0.8, opacity: 0.7, fillColor: '#72aa95', fillOpacity: 0.18 };
type BasinAsset = { cells: Record<string, [number,number][]>; source_modified_at: string; source_url: string; license: string };
let basinCache: BasinAsset | null = null;

export default function CityMap({ gridSize, onLoad, selected, onSelect, counts, reportLayer, predictionScores, showBasins = false }: { gridSize: number; onLoad: (count: number) => void; selected: SelectedCell | null; onSelect: (cell: SelectedCell | null) => void; counts: Record<string, number> | null; reportLayer: boolean; predictionScores: Record<string, number> | null; showBasins?: boolean }) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const boundsRef = useRef<L.LatLngBounds | null>(null);
  const layerRef = useRef<L.GeoJSON | null>(null);
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [tileError, setTileError] = useState(false);
  const [basinInfo, setBasinInfo] = useState('');
  useEffect(() => {
    const map=mapRef.current;
    if (!map || !showBasins || !selected || gridSize !== 500) { setBasinInfo(''); return; }
    const controller=new AbortController();
    let markers:L.LayerGroup | undefined;
    setBasinInfo('Loading catch-basin locations…');
    const request=basinCache ? Promise.resolve(basinCache) : fetch(`${import.meta.env.BASE_URL}infrastructure/catch-basins.json`,{signal:controller.signal}).then(response=>{if(!response.ok)throw new Error('Inventory unavailable');return response.json() as Promise<BasinAsset>;});
    request.then(data=>{
      if(controller.signal.aborted)return;
      if(!data.cells || typeof data.source_modified_at!=='string')throw new Error('Invalid inventory');
      basinCache=data;
      const points=data.cells[selected.id] ?? [];
      markers=L.layerGroup(points.map(([lon,lat])=>L.circleMarker([lat,lon],{radius:4,color:'#fff',weight:1.5,fillColor:'#2263a2',fillOpacity:1,interactive:false}))).addTo(map);
      setBasinInfo(`${points.length} catch-basin records · approximate locations · snapshot ${data.source_modified_at.slice(0,10)}`);
    }).catch(()=>{if(!controller.signal.aborted)setBasinInfo('Catch-basin locations unavailable. Toggle the layer to retry.');});
    return()=>{controller.abort();if(markers)map.removeLayer(markers);};
  },[selected,showBasins,gridSize,status]);

  useEffect(() => {
    if (!container.current) return;
    const map = L.map(container.current, { scrollWheelZoom: false, zoomControl: false, minZoom: 10, maxZoom: 18 }).setView([49.25, -123.12], 11);
    mapRef.current = map;
    L.control.zoom({ position: 'bottomright' }).addTo(map);
    L.control.scale({ imperial: false, position: 'bottomleft' }).addTo(map);
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19,
    }).on('tileerror', () => setTileError(true)).addTo(map);
    const observer = new ResizeObserver(() => {
      map.invalidateSize();
      if (boundsRef.current) map.fitBounds(boundsRef.current, { padding: [24, 24], animate: false });
    });
    observer.observe(container.current);
    return () => { observer.disconnect(); boundsRef.current = null; mapRef.current = null; map.remove(); };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const controller = new AbortController();
    let layer: L.GeoJSON | undefined;
    setStatus('loading');
    boundsRef.current = null;
    const loadGrid = gridCache.has(gridSize)
      ? Promise.resolve(gridCache.get(gridSize)!)
      : fetch(`${import.meta.env.BASE_URL}grids/vancouver_${gridSize}m.geojson`, { signal: controller.signal })
      .then(response => { if (!response.ok) throw new Error('Grid request failed'); return response.json() as Promise<FeatureCollection>; })
    loadGrid.then(data => {
        if (controller.signal.aborted) return;
        gridCache.set(gridSize, data);
        const grid = L.geoJSON(data, { style: cellStyle, onEachFeature: (feature, cellLayer) => {
          cellLayer.on('click', () => onSelect({ id: feature.properties.grid_id, size: feature.properties.grid_size_metres, area: feature.properties.area_m2 }));
          cellLayer.on('add', () => {
            const element = (cellLayer as L.Path).getElement();
            if (!element) return;
            element.setAttribute('role', 'button');
            element.setAttribute('aria-label', `Select cell ${feature.properties.grid_id}`);
            element.setAttribute('tabindex', '0');
            element.addEventListener('keydown', event => {
              const key = (event as KeyboardEvent).key;
              if (key === 'Enter' || key === ' ') { event.preventDefault(); event.stopPropagation(); cellLayer.fire('click'); }
            });
          });
        } }).addTo(map);
        layer = grid;
        layerRef.current = grid;
        boundsRef.current = grid.getBounds();
        map.fitBounds(boundsRef.current, { padding: [24, 24], animate: false });
        onLoad(data.features.length);
        setStatus('ready');
      })
      .catch(() => { if (!controller.signal.aborted) setStatus('error'); });
    return () => { controller.abort(); if (layer) map.removeLayer(layer); layerRef.current = null; };
  }, [gridSize, onLoad, onSelect]);

  useEffect(() => {
    layerRef.current?.eachLayer(layer => {
      const cell = layer as L.Polygon;
      const active = cell.feature?.properties?.grid_id === selected?.id;
      const id = String(cell.feature?.properties?.grid_id);
      const count = counts ? counts[id] ?? 0 : null;
      const score = predictionScores?.[id];
      const style = predictionScores !== null
        ? { ...cellStyle, fillColor: score === undefined ? '#edf1e9' : score >= Math.max(...Object.values(predictionScores!))*.85 ? '#b26446' : score >= Math.max(...Object.values(predictionScores!))*.65 ? '#cf9560' : '#dbbb77', fillOpacity: score === undefined ? 0.08 : 0.85, weight: score === undefined ? 0.5 : 2 }
        : reportLayer && count !== null
        ? { ...cellStyle, fillColor: reportColour(count), fillOpacity: count === 0 ? 0.22 : 0.8 }
        : cellStyle;
      cell.setStyle(active ? { ...style, color: '#172f29', weight: 3, opacity: 1 } : style);
      cell.getElement()?.setAttribute('aria-pressed', String(active));
      cell.getElement()?.setAttribute('aria-description', predictionScores !== null ? score === undefined ? 'Outside the top-five highlighted areas' : `Experimental report estimate ${(score*100).toFixed(1)} percent. Not a flood probability.` : count === null ? 'Report counts unavailable' : `${count} historical reports in the selected period`);
      cell.unbindTooltip();
      if (reportLayer || predictionScores !== null) {
        const tooltip = document.createElement('span');
        tooltip.textContent = `${id} · ${predictionScores !== null ? score === undefined ? 'Outside top five' : `Experimental report estimate ${(score*100).toFixed(1)}%` : count === null ? 'Reports unavailable' : `${count} report${count === 1 ? '' : 's'}`}`;
        cell.bindTooltip(tooltip, { sticky: true });
      }
      if (active) cell.bringToFront();
    });
  }, [selected, counts, reportLayer, predictionScores, status]);

  useEffect(() => {
    if (!selected || predictionScores === null) return;
    layerRef.current?.eachLayer(layer => {
      const cell = layer as L.Polygon;
      if (cell.feature?.properties?.grid_id === selected.id) mapRef.current?.panTo(cell.getBounds().getCenter());
    });
  }, [selected, status]);

  return <div className="map-stage real-map-stage">
    <div ref={container} className="leaflet-map" role="region" aria-label="Vancouver grid map. Use arrow keys to pan and plus or minus to zoom." />
    <span className="map-preview-label">{gridSize.toLocaleString()} M GRID · {predictionScores !== null ? 'EXPERIMENTAL · HISTORICAL MODEL' : reportLayer ? counts ? 'HISTORICAL REPORT COUNTS' : 'COUNTS UNAVAILABLE' : 'GRID COVERAGE'}</span>
    <button className="fit-map-button" disabled={status !== 'ready'} onClick={() => { if (boundsRef.current) mapRef.current?.fitBounds(boundsRef.current, { padding: [24, 24] }); }}>Fit Vancouver</button>
    {status !== 'ready' && <div className="map-status" role="status">{status === 'loading' ? 'Loading the Vancouver grid…' : 'The grid could not load. Refresh to try again.'}</div>}
    {tileError && <div className="tile-warning" role="status">Street map unavailable. The grid still works.</div>}
    {showBasins && <div className="basin-map-note" role="status">{selected ? basinInfo : 'Select an area to show catch-basin locations.'}<br/><a href="https://opendata.vancouver.ca/explore/dataset/sewer-catch-basins/information/" target="_blank" rel="noreferrer">City of Vancouver · Open Government Licence</a></div>}
  </div>;
}
