"""Production routing and readiness checks against the built dashboard."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.production import app


class ProductionTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_dashboard_and_grid_are_served_without_vite(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="root"', response.text)
        self.assertEqual(len(self.client.get("/grids/vancouver_500m.geojson").json()["features"]), 604)

    def test_preparation_routes_and_unknown_api_are_not_exposed(self):
        self.assertEqual(self.client.post("/data/clean-311").status_code, 405)
        response = self.client.get("/api/unknown")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Not found"})

    def test_readiness_requires_model_and_built_assets(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/ready").status_code, 200)
        with patch("app.services.prediction.artifact", side_effect=ValueError("broken artifact")):
            self.assertEqual(self.client.get("/ready").status_code, 503)


if __name__ == "__main__":
    unittest.main()
