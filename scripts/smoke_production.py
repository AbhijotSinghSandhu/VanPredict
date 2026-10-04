"""Run the same smoke checks against localhost or a deployed HTTPS service."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from time import perf_counter
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Base URL, for example https://vanpredict.onrender.com")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    def request(path, body=None, expected=200):
        payload = None if body is None else json.dumps(body).encode()
        req = Request(base + path, data=payload, headers={"Content-Type": "application/json"})
        started = perf_counter()
        try:
            response = urlopen(req, timeout=90)
        except HTTPError as error:
            response = error
        with response:
            data = response.read()
            assert response.status == expected, (path, response.status, data[:300])
            print(f"PASS {path}: {response.status} ({perf_counter()-started:.2f}s)")
            return json.loads(data) if "json" in response.headers.get("Content-Type", "") or path.endswith(".geojson") else data

    request("/")
    request("/health")
    request("/ready")
    grid = request("/grids/vancouver_500m.geojson")
    assert len(grid["features"]) == 604
    for size in (200, 500, 1000):
        result = request(f"/api/v1/reports?grid_size={size}")
        assert result["total_reports"] == 40560
        assert sum(result["counts"].values()) == result["mapped_reports"]
    rain = request("/api/v1/rainfall?start=2025-06-04&end=2025-07-04&grid_size=500")
    assert rain["city"]["total_mm"] == 35.21
    assert rain["city"]["covered_days"] == 31
    cell_rain = request("/api/v1/rainfall?start=2025-06-04&end=2025-07-04&grid_size=500&cell_id=g500_980_10918")
    assert cell_rain["cell"]["total_mm"] == 28
    request("/api/v1/model-status")
    result = request("/api/v1/predict", {"prediction_time": "2025-06-04T06:00:00-07:00", "top_k": 5})
    assert len(result["areas"]) == 604 and len(result["top_areas"]) == 5
    probabilities = [area["report_probability"] for area in result["areas"]]
    assert all(0 <= value <= 1 for value in probabilities)
    assert probabilities == sorted(probabilities, reverse=True)
    assert result["uses_forecast_rainfall"] is False
    request("/api/v1/predict", {"grid_size_m": 200}, expected=422)
    request("/api/v1/predict", {"prediction_time": "2026-10-04T06:00:00-07:00"}, expected=422)
    request("/api/unknown", expected=404)
    request("/data/build-grid", {}, expected=405)
    request("/api/v1/reports?grid_size=123", expected=422)
    request("/api/v1/rainfall?start=2025-07-04&end=2025-06-04", expected=422)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(request, ["/api/v1/reports?grid_size=500"] * 8))
    print("Production smoke checks passed, including eight concurrent report requests.")


if __name__ == "__main__":
    main()
