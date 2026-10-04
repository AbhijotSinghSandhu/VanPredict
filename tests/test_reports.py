"""Source reconciliation and date-boundary checks for historical aggregates."""
import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from app.services.reports import summarize
from app.routers.reports import reports


class ReportTests(unittest.TestCase):
    def test_all_grid_sizes_reconcile_with_source(self):
        source = pd.read_csv(ROOT / 'data/processed/311_gridded.csv')
        for size in (200, 500, 1000):
            with self.subTest(size=size):
                result = summarize(size)
                self.assertEqual(result['total_reports'], len(source))
                self.assertEqual(sum(result['counts'].values()), result['mapped_reports'])
                self.assertEqual(result['mapped_reports'] + result['unmapped_reports'], len(source))
                self.assertEqual(result['unmapped_reports'], 0)

    def test_same_day_includes_both_boundaries_and_matches_cells(self):
        source = pd.read_csv(ROOT / 'data/processed/311_gridded.csv')
        expected = source.loc[source.date == '2009-06-24']
        result = summarize(500, date(2009, 6, 24), date(2009, 6, 24))
        self.assertEqual(result['total_reports'], len(expected))
        self.assertEqual(result['counts'], expected.groupby('grid_id').size().to_dict())

    def test_outside_dates_show_empty_and_coverage_notice(self):
        result = summarize(500, date(2000, 1, 1), date(2000, 1, 2))
        self.assertEqual(result['total_reports'], 0)
        self.assertEqual(result['counts'], {})
        self.assertTrue(result['partial_coverage'])

    def test_invalid_queries(self):
        for args in [(date(2024, 3, 2), date(2024, 3, 1), 500),
                     (date(2024, 1, 1), None, 500), (None, None, 300)]:
            with self.assertRaises(HTTPException) as caught:
                reports(*args)
            self.assertEqual(caught.exception.status_code, 422)


if __name__ == '__main__':
    unittest.main()
