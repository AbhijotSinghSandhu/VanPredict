import sys
import unittest
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train_spatial_model import SPLITS, logits, load_split
from src.spatial_training_data import TRAINING_FEATURES


class SpatialModelTests(unittest.TestCase):
    def test_target_partitions_do_not_overlap(self):
        self.assertLess(SPLITS['training'][1], SPLITS['calibration'][0])
        self.assertLess(SPLITS['calibration'][1], SPLITS['test'][0])

    def test_extreme_scores_can_be_calibrated(self):
        self.assertTrue(np.isfinite(logits(np.array([0, 1e-9, .5, 1]))).all())

    @unittest.skipUnless((ROOT / 'models/spatial_nextday_v1.joblib').exists(), 'Train the local artifact first')
    def test_saved_artifact_accepts_real_features(self):
        artifact = joblib.load(ROOT / 'models/spatial_nextday_v1.joblib')
        self.assertEqual(artifact['features'], TRAINING_FEATURES)
        x, y, _, dates = load_split('2025-06-04', '2025-06-04')
        self.assertEqual(len(x), 604)
        self.assertTrue((dates == '2025-06-04').all())
        scores = artifact['calibration'].predict_proba(logits(artifact['model'].predict_proba(x)[:, 1]))[:, 1]
        self.assertEqual(scores.shape, y.shape)
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())


if __name__ == '__main__':
    unittest.main()
