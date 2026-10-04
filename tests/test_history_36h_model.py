import unittest
import numpy as np
from src.train_history_36h import evaluate, logits, SPLITS, ROOT


class HistoryModelTests(unittest.TestCase):
    def test_shortlists_are_per_issue_and_use_stable_grid_ties(self):
        grids = np.array([f'g{i:02}' for i in range(25)]*2)
        times = np.array(['a']*25+['b']*25)
        labels = np.zeros(50,dtype=int)
        labels[[0,25]]=1
        result = evaluate(labels,np.full(50,.04),times,grids)
        self.assertAlmostEqual(result['shortlists']['5']['precision'],.2)
        self.assertEqual(result['shortlists']['5']['recall'],1)
        self.assertAlmostEqual(result['shortlists']['20']['precision'],.05)

    def test_calibration_handles_extremes_and_invalid_scores_reject(self):
        self.assertTrue(np.isfinite(logits(np.array([0.,1.]))).all())
        with self.assertRaisesRegex(ValueError,'Invalid'):
            evaluate(np.array([0,1]),np.array([0,np.nan]),np.array(['a','a']),np.array(['g1','g2']))

    def test_development_is_after_training_and_calibration(self):
        self.assertLess(SPLITS['training'][1],SPLITS['calibration'][0])
        self.assertLess(SPLITS['calibration'][1],SPLITS['development'][0])

    @unittest.skipUnless((ROOT/'models/history_36h_v1.joblib').exists(), 'Train artifact first')
    def test_saved_model_matches_feature_contract(self):
        import joblib
        import pandas as pd
        from src.training_36h import FEATURES
        artifact = joblib.load(ROOT/'models/history_36h_v1.joblib')
        self.assertEqual(artifact['features'],FEATURES)
        sample = pd.read_csv(ROOT/'data/processed/spatial_training/history_36h.csv.gz',nrows=604)
        scores = artifact['calibration'].predict_proba(logits(
            artifact['model'].predict_proba(sample[FEATURES].to_numpy(dtype='float32'))[:,1]))[:,1]
        self.assertEqual(len(scores),604)
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores>=0)&(scores<=1)).all())


if __name__ == '__main__':
    unittest.main()
