import os
import unittest
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from visualize_h5 import (
    get_latest_file,
    get_latest_h5_file,
    get_latest_training_csv,
    get_feature_estimates,
    load_model_and_generate_predictions,
    get_h5_plot,
    H5_DIRECTORY,
    CSV_DIRECTORY
)

class TestVisualizeH5(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        for f in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, f))
        os.rmdir(self.test_dir)

    def test_get_latest_file_returns_none_if_no_files(self):
        result = get_latest_file(self.test_dir, '*.csv')
        self.assertIsNone(result)

    def test_get_latest_file_returns_latest(self):
        file1 = os.path.join(self.test_dir, 'file1.csv')
        file2 = os.path.join(self.test_dir, 'file2.csv')
        with open(file1, 'w') as f:
            f.write('test1')
        with open(file2, 'w') as f:
            f.write('test2')
        os.utime(file1, (1, 1))
        os.utime(file2, (2, 2))

        result = get_latest_file(self.test_dir, '*.csv')
        self.assertEqual(result, file2)

    @patch('visualize_h5.get_latest_file')
    def test_get_latest_h5_file(self, mock_get):
        mock_get.return_value = '/path/to/latest.h5'
        self.assertEqual(get_latest_h5_file(), '/path/to/latest.h5')

    @patch('visualize_h5.get_latest_file')
    def test_get_latest_training_csv(self, mock_get):
        mock_get.return_value = '/path/to/latest.csv'
        self.assertEqual(get_latest_training_csv(), '/path/to/latest.csv')

    @patch('visualize_h5.get_latest_training_csv')
    def test_get_feature_estimates_returns_means(self, mock_csv):
        df = pd.DataFrame({"y1": [1, 2, 3], "y2": [4, 5, 6]})
        temp_csv = os.path.join(self.test_dir, 'test.csv')
        df.to_csv(temp_csv, index=False)
        mock_csv.return_value = temp_csv

        y1, y2 = get_feature_estimates()
        self.assertEqual(y1, 2.0)
        self.assertEqual(y2, 5.0)

    @patch('visualize_h5.tf.keras.models.load_model')
    @patch('visualize_h5.get_latest_h5_file')
    @patch('visualize_h5.get_feature_estimates')
    def test_load_model_and_generate_predictions(self, mock_est, mock_h5, mock_load):
        mock_h5.return_value = '/mock/model.h5'
        mock_est.return_value = (5.0, 5.0)

        dummy_model = MagicMock()
        dummy_model.predict.return_value = np.random.rand(900, 1)
        mock_load.return_value = dummy_model

        y1, y2, z = load_model_and_generate_predictions()
        self.assertIsNotNone(y1)
        self.assertIsNotNone(y2)
        self.assertIsNotNone(z)
        self.assertEqual(y1.shape, y2.shape)
        self.assertEqual(y1.shape, z.shape)

    @patch('visualize_h5.load_model_and_generate_predictions')
    def test_get_h5_plot_returns_html(self, mock_load):
        y1 = y2 = np.linspace(0, 1, 10)
        Y1, Y2 = np.meshgrid(y1, y2)
        Z = np.random.rand(10, 10)
        mock_load.return_value = (Y1, Y2, Z)

        html = get_h5_plot()
        self.assertIn('<div', html)
        self.assertIn('Model Prediction', html)

    @patch('visualize_h5.load_model_and_generate_predictions')
    def test_get_h5_plot_handles_failure(self, mock_load):
        mock_load.return_value = (None, None, None)
        result = get_h5_plot()
        self.assertIn('Error', result)

if __name__ == '__main__':
    unittest.main()