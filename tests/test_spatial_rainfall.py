import unittest

import numpy as np
import pandas as pd

from backend.app.services.spatial_rainfall import FEATURES, create_features
from notebooks.build_spatial_rainfall import geometry_centroid, grid_mapping, ROOT


def raw_frame(start="2025-10-20", end="2025-11-05 23:00"):
    times = pd.date_range(start, end, freq="h", tz="UTC")
    return pd.DataFrame({"datetime": times, "rainfall_mm": np.ones(len(times)),
                         "weather_cell_id": "test"})


class SpatialRainfallTests(unittest.TestCase):
    def test_calendar_days_and_daylight_saving(self):
        hourly, daily = create_features(raw_frame())
        day = daily.set_index("local_date").loc["2025-11-02"]
        self.assertEqual(day.observed_hours, 25)
        self.assertEqual(day.expected_hours, 25)
        self.assertEqual(day.rainfall_today, 25)
        self.assertEqual(day.rain_duration_hours, 25)
        self.assertEqual(day.rainfall_previous_day, 24)
        self.assertEqual(day.rainfall_previous_3d, 72)
        self.assertEqual(day.rainfall_previous_7d, 168)
        self.assertEqual(day.max_rain_6h, 6)
        self.assertEqual(day.max_rain_24h, 24)
        self.assertEqual(day.heavy_rain_24h, 0)
        self.assertTrue(day.features_ready)
        spring = create_features(raw_frame("2025-03-01", "2025-03-12"))[1]
        day = spring.set_index("local_date").loc["2025-03-09"]
        self.assertEqual(day.observed_hours, 23)
        self.assertEqual(day.expected_hours, 23)
        self.assertEqual(day.rain_duration_hours, 23)

    def test_midnight_interval_and_thresholds(self):
        raw = raw_frame()
        raw.rainfall_mm = 0.0
        # Midnight November 1 local time closes the final October 31 hour.
        raw.loc[raw.datetime.eq(pd.Timestamp("2025-11-01 07:00", tz="UTC")), "rainfall_mm"] = 25
        hourly, daily = create_features(raw)
        before = hourly.set_index("datetime").loc[pd.Timestamp("2025-11-01 07:00", tz="UTC")]
        after = hourly.set_index("datetime").loc[pd.Timestamp("2025-11-01 08:00", tz="UTC")]
        self.assertEqual(before.local_date, "2025-10-31")
        self.assertEqual(before.rainfall_today, 25)
        self.assertEqual(after.local_date, "2025-11-01")
        self.assertEqual(after.rainfall_today, 0)
        self.assertEqual(after.rainfall_previous_day, 25)
        self.assertEqual(after.max_rain_1h, 0)
        self.assertEqual(after.max_rain_6h, 25)
        self.assertEqual(after.heavy_rain_6h, 1)
        self.assertEqual(after.heavy_rain_24h, 1)
        self.assertEqual(before.heavy_rain_1h, 1)
        self.assertEqual(daily.set_index("local_date").loc["2025-10-31"].rain_duration_hours, 1)

    def test_hourly_features_cannot_use_future_rain(self):
        raw = raw_frame()
        cutoff = pd.Timestamp("2025-11-01 20:00", tz="UTC")
        original = create_features(raw)[0]
        changed = raw.copy()
        changed.loc[changed.datetime.gt(cutoff), "rainfall_mm"] = 100
        updated = create_features(changed)[0]
        pd.testing.assert_frame_equal(
            original.loc[original.datetime.le(cutoff), FEATURES],
            updated.loc[updated.datetime.le(cutoff), FEATURES],
        )
        final_row = original.loc[original.datetime.eq(cutoff)].iloc[0]
        self.assertEqual(final_row.rainfall_today, 13)

    def test_incomplete_antecedent_not_silently_filled(self):
        hourly, daily = create_features(raw_frame())
        self.assertFalse(daily.iloc[0].day_complete)
        self.assertTrue(pd.isna(daily.iloc[1].rainfall_previous_day))
        self.assertTrue(pd.isna(daily.iloc[7].rainfall_previous_7d))
        self.assertFalse(hourly.iloc[0].features_ready)
        self.assertTrue(daily.iloc[8].features_ready)

    def test_missing_duplicate_or_invalid_hours_rejected(self):
        raw = raw_frame()
        for invalid in (raw.drop(index=10), pd.concat([raw, raw.iloc[[10]]])):
            with self.assertRaises(ValueError):
                create_features(invalid)
        for value in (-1, np.nan):
            invalid = raw.copy()
            invalid.loc[10, "rainfall_mm"] = value
            with self.assertRaises(ValueError):
                create_features(invalid)

    def test_centroids_and_existing_grid_mapping(self):
        polygon = {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]}
        np.testing.assert_allclose(geometry_centroid(polygon), [1, 1])
        mapping = grid_mapping(ROOT / "data/processed/vancouver_grid.geojson")
        self.assertEqual(len(mapping), 604)
        self.assertFalse(mapping.grid_id.duplicated().any())
        self.assertTrue(mapping.grid_size_m.eq(500).all())
        self.assertTrue((abs(mapping.weather_latitude - mapping.centroid_latitude) <= 0.125).all())
        self.assertTrue((abs(mapping.weather_longitude - mapping.centroid_longitude) <= 0.125).all())


if __name__ == "__main__":
    unittest.main()
