import unittest

import pandas as pd

from backend.app.services.spatial_rainfall import FEATURES
from src.spatial_training_data import attach_next_day_labels


class SpatialTrainingTests(unittest.TestCase):
    def setUp(self):
        self.panel = pd.DataFrame([
            {"grid_id": grid, "local_date": day, **dict.fromkeys(FEATURES, 0.0)}
            for grid in ("a", "b") for day in ("2025-11-01", "2025-11-02")
        ])
        self.reports = pd.DataFrame([
            {"grid_id": "a", "service_request_open_timestamp": "2025-11-02T06:00:00Z",
             "service_request_type": "Street Surface Water Flooding Case"},
            {"grid_id": "a", "service_request_open_timestamp": "2025-11-02T12:00:00Z",
             "service_request_type": "Catch Basin Concern Case"},
            {"grid_id": "b", "service_request_open_timestamp": "2025-11-02T12:00:00Z",
             "service_request_type": "Sewer Backup Case"},
        ])

    def test_labels_use_next_local_day_and_matching_grid(self):
        table = attach_next_day_labels(self.panel, self.reports, "2025-11-01", "2025-11-03")
        indexed = table.set_index(["grid_id", "local_date"])
        self.assertEqual(indexed.loc[("a", "2025-11-01"), "next_day_report_count"], 1)
        self.assertEqual(indexed.loc[("a", "2025-11-02"), "next_day_report_count"], 0)
        self.assertEqual(indexed.loc[("b", "2025-11-01"), "is_drainage_problem_next_day"], 0)
        self.assertEqual(len(table), 4)

    def test_unavailable_target_days_are_not_zero_labels(self):
        with self.assertRaisesRegex(ValueError, "outside report coverage"):
            attach_next_day_labels(self.panel, self.reports, "2025-11-01", "2025-11-02")

    def test_duplicate_features_and_missing_rain_are_rejected(self):
        with self.assertRaises(ValueError):
            attach_next_day_labels(pd.concat([self.panel, self.panel]), self.reports, "2025-11-01", "2025-11-03")
        invalid = self.panel.copy()
        invalid.loc[0, "rainfall_today"] = float("nan")
        with self.assertRaises(ValueError):
            attach_next_day_labels(invalid, self.reports, "2025-11-01", "2025-11-03")


if __name__ == "__main__":
    unittest.main()
