# History-only 36-hour dataset

`src/training_36h.py` creates one row per 500 m cell at 06:00 Vancouver time each day. The target is at least one Street Surface Water Flooding or Catch Basin Concern report in [issue time, issue time + 36 elapsed hours). UTC arithmetic handles daylight-saving changes. Windows overlap; counts cannot be compared directly with unique report totals.

Inputs are location, calendar, past 24/72/168-hour rainfall, past 24/168/720-hour matching reports, and cumulative matching report frequency per day since the declared coverage start. No future rain or target counts enter FEATURES. Forecast inputs and model training are intentionally deferred. Report history uses timestamps strictly before issue minus the assumed publication lag (default 24 hours). Windows require 30 days of report history before that cutoff.

Rainfall uses hourly interval ends in (issue minus duration, issue], rejecting missing hours. ERA5 is retrospective and can be revised; this dataset does not establish that these exact weather values were available at historical issue times. A fixed report lag is also not proof of historical publication availability. Neither report archive completeness nor update delays have been audited. Declare coverage explicitly; never silently label unavailable periods as zero reports. Original historical snapshots would be needed to audit revisions and availability.

Example research sample (coverage is provisionally based on the observed archive date span):

```powershell
.venv/Scripts/python.exe src/training_36h.py --start-date 2025-06-04 --end-date 2025-07-04 --coverage-start 2009-06-24T00:00:00-07:00 --coverage-end 2026-10-03T00:00:00-07:00
.venv/Scripts/python.exe -m unittest tests.test_training_36h tests.test_spatial_training_data
```

The default gzip CSV is `data/processed/spatial_training/history_36h.csv.gz`; a metadata sidecar records assumptions and feature order. The sample has 18,724 rows (31 × 604) and 68 positive cell-windows. These are two target categories in overlapping windows, not the dashboard's 82 total historical drainage requests.

Before training, choose chronological development partitions and use `partition()` to exclude any target window crossing a boundary. Reserve fresh confirmation data rather than treating the previously inspected test period as untouched. Use FEATURES explicitly so label/metadata columns never enter training. Evaluate shortlist precision and recall at 5/10/20 cells, temporal stability and calibration against a historical-frequency baseline. Keep 36-hour probabilities separate from next-day model outputs.

Next: audit coverage and publication delay; build longer historical datasets; train the history-only baseline; acquire as-issued historical forecasts or collect live forecast snapshots before testing forecast improvements. Daily 06:00 examples do not yet validate arbitrary-hour live requests. No API or existing model changes are included in this milestone.
