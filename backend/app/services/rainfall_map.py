"""Interactive rain layer on the existing 500 m grid, including offline data."""

import json

import pandas as pd

from .spatial_rainfall import DATASET_DIR, FEATURES

LABELS = [
    "Rainfall today", "Maximum 1-hour rain", "Maximum 6-hour rain",
    "Maximum 24-hour rain", "Heavy rain in 1 hour", "Heavy rain in 6 hours",
    "Heavy rain in 24 hours", "Hours with rain", "Previous day's rain",
    "Previous 3 days' rain", "Previous 7 days' rain",
]


def rainfall_map_data(dataset_dir=DATASET_DIR):
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    mapping = pd.read_csv(dataset_dir / "grid_weather_mapping.csv")
    hourly = pd.read_csv(dataset_dir / "weather_hourly_features.csv.gz")
    daily = pd.read_csv(dataset_dir / "weather_daily_features.csv.gz")
    cells = {}
    for cell_id, frame in daily.groupby("weather_cell_id", sort=False):
        cells[cell_id] = {
            "daily": dict(zip(frame.local_date, frame[FEATURES].round(3).values.tolist())),
            "hourly": {},
        }
    # Use Timestamp.timestamp rather than integer datetime resolution assumptions.
    hourly["epoch_seconds"] = [int(stamp.timestamp()) for stamp in pd.to_datetime(hourly.datetime, utc=True)]
    for (cell_id, day), frame in hourly.groupby(["weather_cell_id", "local_date"], sort=False):
        cells[cell_id]["hourly"][day] = [
            [int(epoch), *values] for epoch, values in zip(
                frame.epoch_seconds, frame[FEATURES].round(3).values.tolist()
            )
        ]
    return {
        "start": manifest["start_local_date"], "end": manifest["end_local_date"],
        "fields": FEATURES, "labels": LABELS, "cells": cells,
        "mapping": dict(zip(mapping.grid_id, mapping.weather_cell_id)),
    }


def add_rainfall_layer(map_view, collection, dataset_dir=DATASET_DIR):
    import folium
    from branca.element import MacroElement, Template

    payload = rainfall_map_data(dataset_dir)
    grid_ids = {feature["properties"]["grid_id"] for feature in collection["features"]}
    if grid_ids != set(payload["mapping"]):
        raise ValueError("Rainfall mapping does not match the current grid; rebuild the rainfall dataset")
    rainfall = folium.GeoJson(
        collection, name="Rainfall — hourly / daily", show=True,
        style_function=lambda feature: {"fillColor": "#dbeafe", "color": "#475569",
                                        "weight": 0.6, "fillOpacity": 0.65},
        highlight_function=lambda feature: {"weight": 2, "color": "#0f172a"},
    ).add_to(map_view)
    controller = MacroElement()
    controller._name = "RainfallGridControl"
    # Numeric arrays keep all 17 years of hourly data small enough to embed.
    # Data comes from validated local artifacts; escape HTML script terminators.
    controller.payload = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    controller.layer_name = rainfall.get_name()
    controller.map_name = map_view.get_name()
    controller.template = Template(r"""
{% macro script(this, kwargs) %}
(function () {
  const data = {{ this.payload | safe }};
  const layer = {{ this.layer_name }};
  const map = {{ this.map_name }};
  const units = i => i >= 4 && i <= 6 ? '' : i === 7 ? ' hours' : ' mm';
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const format = (value, i) => value == null ? 'Unavailable' : i >= 4 && i <= 6 ? (value ? 'Yes' : 'No') : Number(value).toFixed(i === 7 ? 0 : 1) + units(i);
  const timeFormat = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'America/Vancouver', hour: '2-digit', minute: '2-digit',
    hourCycle: 'h23', timeZoneName: 'shortOffset'
  });
  let panel;
  let selectedDate = data.start <= '2025-11-01' && data.end >= '2025-11-01' ? '2025-11-01' : data.end;
  const control = L.control({position: 'topleft'});
  control.onAdd = function () {
    panel = L.DomUtil.create('section', 'rainfall-control');
    panel.style.cssText = 'width:285px;max-width:calc(100vw - 85px);padding:14px 16px;background:white;border-radius:10px;box-shadow:0 2px 14px #0003;font:13px/1.5 system-ui;color:#172554;max-height:70vh;overflow:auto';
    panel.innerHTML = `<strong style="font-size:17px">Rainfall on your 500 m grid</strong>
      <p style="margin:5px 0 12px;color:#475569">Choose a day, then click a cell.</p>
      <label style="display:block">Vancouver rainfall date<br><input class="rain-date" type="date" min="${data.start}" max="${data.end}" value="${selectedDate}" style="width:100%;padding:5px"></label>
      <label style="display:block;margin-top:8px">Time view<br><select class="rain-mode" style="width:100%;padding:5px"><option value="daily">Completed day</option><option value="hourly">Hourly — through selected time</option></select></label>
      <label class="rain-hour-label" style="display:none;margin-top:8px">Completed hour ending at<br><select class="rain-hour" style="width:100%;padding:5px"></select></label>
      <label style="display:block;margin-top:8px">Colour cells by<br><select class="rain-field" style="width:100%;padding:5px">${data.labels.map((label,i) => `<option value="${i}">${escape(label)}</option>`).join('')}</select></label>
      <div class="rain-scale" style="margin-top:12px"></div>
      <p class="rain-note" style="margin:8px 0;color:#475569"></p>
      <p style="margin:0;font-size:11px;color:#64748b">ERA5 • 2 shared weather cells • liquid rain<br>Previous days exclude the selected day.<br>Hourly amounts cover the preceding hour; midnight closes the previous day's final hour.</p>`;
    L.DomEvent.disableClickPropagation(panel);
    L.DomEvent.disableScrollPropagation(panel);
    return panel;
  };
  control.addTo(map);
  const dateInput = panel.querySelector('.rain-date');
  const modeInput = panel.querySelector('.rain-mode');
  const hourInput = panel.querySelector('.rain-hour');
  const fieldInput = panel.querySelector('.rain-field');
  function rebuildHours() {
    selectedDate = dateInput.value;
    const rows = Object.values(data.cells)[0].hourly[selectedDate] || [];
    hourInput.innerHTML = rows.map(row => `<option value="${row[0]}">${escape(timeFormat.format(new Date(row[0] * 1000)))}</option>`).join('');
    if (rows.length) hourInput.value = String(rows[rows.length - 1][0]);
  }
  function valuesFor(cell) {
    if (modeInput.value === 'daily') return cell.daily[selectedDate];
    const row = (cell.hourly[selectedDate] || []).find(row => row[0] === Number(hourInput.value));
    return row ? row.slice(1) : null;
  }
  function update() {
    const metric = Number(fieldInput.value);
    const rows = Object.values(data.cells).map(valuesFor).filter(Boolean);
    const maximum = Math.max(0, ...rows.map(row => row[metric]));
    const color = value => {
      if (value == null) return '#cbd5e1';
      if (metric >= 4 && metric <= 6) return value ? '#1d4ed8' : '#eff6ff';
      if (!maximum) return '#eff6ff';
      const weight = Math.min(1, Math.max(0, value / maximum));
      return `rgb(${Math.round(239 - 210*weight)},${Math.round(246 - 168*weight)},${Math.round(255 - 39*weight)})`;
    };
    panel.querySelector('.rain-hour-label').style.display = modeInput.value === 'hourly' ? 'block' : 'none';
    panel.querySelector('.rain-scale').innerHTML = `<div style="height:10px;border-radius:5px;background:linear-gradient(to right,#eff6ff,#1d4ed8)"></div><div style="display:flex;justify-content:space-between;margin-top:3px"><span>${metric >= 4 && metric <= 6 ? 'No' : '0' + units(metric)}</span><span>${metric >= 4 && metric <= 6 ? 'Yes' : format(maximum, metric)}</span></div>`;
    panel.querySelector('.rain-note').textContent = rows.length ?
      (modeInput.value === 'daily' ? 'Complete-day historical summary.' : 'Uses completed rainfall hours through the selected time.') :
      'No rainfall data for this date. Choose a date within the available period.';
    // Folium restores GeoJSON's base style after hover; keep that base style
    // synchronized with the current date and metric as well as visible cells.
    layer.options.style = feature => {
      const values = valuesFor(data.cells[data.mapping[feature.properties.grid_id]]);
      return {fillColor: color(values ? values[metric] : null), color: '#475569', weight: 0.6, fillOpacity: 0.65};
    };
    layer.eachLayer(cellLayer => {
      const p = cellLayer.feature.properties;
      const cellId = data.mapping[p.grid_id];
      const values = valuesFor(data.cells[cellId]);
      cellLayer.setStyle({fillColor: color(values ? values[metric] : null), fillOpacity: 0.65});
      const heading = modeInput.value === 'daily' ? selectedDate : `${selectedDate} · hour ending ${timeFormat.format(new Date(Number(hourInput.value)*1000))}`;
      const fields = values ? data.labels.map((label,i) => `<tr><td style="padding:3px 12px 3px 0">${escape(label)}</td><td style="text-align:right;font-weight:600">${format(values[i],i)}</td></tr>`).join('') : '<tr><td>No data for this date</td></tr>';
      cellLayer.bindPopup(`<div style="font:13px/1.5 system-ui"><strong style="font-size:15px">${escape(p.grid_id)}</strong><div style="color:#475569;margin:4px 0 8px">${escape(heading)}</div><table>${fields}</table><hr style="border:0;border-top:1px solid #e2e8f0"><div>Historical reports: ${Number(p.report_count).toLocaleString()}<br>Area: ${Number(p.area_m2).toLocaleString()} m²<br>Weather cell: ${escape(cellId)}<br><small>Rainfall is shared with other grids using this weather cell.</small></div></div>`, {maxWidth:360});
      cellLayer.bindTooltip(`${escape(p.grid_id)}<br>${escape(data.labels[metric])}: ${values ? format(values[metric],metric) : 'Unavailable'}`);
    });
  }
  dateInput.addEventListener('change', () => { rebuildHours(); update(); });
  modeInput.addEventListener('change', update);
  hourInput.addEventListener('change', update);
  fieldInput.addEventListener('change', update);
  rebuildHours();
  update();
})();
{% endmacro %}
""")
    controller._template = controller.template
    # Register after the GeoJSON script so its Leaflet layer already exists.
    map_view.add_child(controller)
    return rainfall
