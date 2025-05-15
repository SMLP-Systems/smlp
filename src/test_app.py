import os
import unittest
import tempfile
import subprocess
from unittest.mock import patch
from app import app, call_smlp_api

class AppTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        self.upload_folder = tempfile.mkdtemp()
        app.config['UPLOAD_FOLDER'] = self.upload_folder

    def tearDown(self):
        for file in os.listdir(self.upload_folder):
            os.remove(os.path.join(self.upload_folder, file))
        os.rmdir(self.upload_folder)

    def test_home(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_explore_get(self):
        response = self.client.get('/explore')
        self.assertEqual(response.status_code, 200)

    def test_explore_invalid_mode(self):
        response = self.client.post('/explore', data={'explore_mode': 'invalid_mode'})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Error: Invalid mode selected", response.data)

    def test_explore_missing_files(self):
        response = self.client.post('/explore', data={'explore_mode': 'certify'})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Error: Missing required dataset or spec file", response.data)

    def test_explore_full_form_submission(self):
        data_file_path = os.path.join(self.upload_folder, "data.csv")
        spec_file_path = os.path.join(self.upload_folder, "spec.txt")

        with open(data_file_path, 'w') as f:
            f.write("sample,data,content")
        with open(spec_file_path, 'w') as f:
            f.write('''{
        "version": "1.2",
        "spec": {
            "targets": ["y1"],
            "constraints": []
        }
    }''')
        with self.client.session_transaction() as session:
            output = session.get('output', '')
            self.assertNotIn('Exception', output)
            self.assertNotIn('Traceback', output)

        with open(data_file_path, 'rb') as data_file, open(spec_file_path, 'rb') as spec_file:
            response = self.client.post('/explore', data={
                'explore_mode': 'optimize',
                'data_file': (data_file, 'data.csv'),
                'spec_file': (spec_file, 'spec.txt'),
                'out_dir_val': './results',
                'pref_val': 'Test113',
                'pareto': 't',
                'resp': 'y',
                'feat': 'x1,x2',
                'model_expr': 'dt',
                'dt_sklearn_max_depth': '5',
                'mrmr_pred': '5',
                'epsilon': '0.01',
                'delta_rel': '0.05',
                'save_model_config': 'yes',
                'plots': 'yes',
                'log_time': 'yes',
                'seed_val': '42',
                'objv_names': 'objv1,objv2',
                'objv_exprs': 'y1>7 and y2<3',
                'additional_command': '-seed 10'
            }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers['Location'].endswith('/results'))

    def test_train_missing_file(self):
        response = self.client.post('/train', data={})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Error: Missing required dataset or spec file", response.data)

    def test_train_valid_submission(self):
        data_file_path = os.path.join(self.upload_folder, "data.csv")
        with open(data_file_path, 'w') as f:
            f.write("sample,data,content")

        with open(data_file_path, 'rb') as data_file:
            response = self.client.post('/train', data={
                'data_file': (data_file, 'data.csv'),
                'out_dir_val': './results'
            }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers['Location'].endswith('/results'))

    def test_call_smlp_api(self):
        result = call_smlp_api(['echo', 'hello'])
        self.assertEqual(result.strip(), 'hello')

    def test_results_route(self):
        plot_dir = os.path.abspath("../regr_smlp/code/images/results")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, "dummy_plot.png")
        with open(plot_path, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')

        with self.client.session_transaction() as sess:
            sess['output'] = 'Test output message'
            sess['use_h5'] = False

        response = self.client.get('/results')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Test output message', response.data)
        self.assertIn(b'dummy_plot.png', response.data)

        os.remove(plot_path)

    def test_serve_plot_route(self):
        plot_dir = os.path.abspath("../regr_smlp/code/images/results")
        os.makedirs(plot_dir, exist_ok=True)
        filename = "test_image.png"
        filepath = os.path.join(plot_dir, filename)

        with open(filepath, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')

        with open(filepath, 'rb') as f:
            response = self.client.get(f'/plots/{filename}')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, 'image/png')

        os.remove(filepath)

    @patch('app.get_h5_plot')
    def test_results_route_with_h5(self, mock_get_h5_plot):
        mock_get_h5_plot.return_value = "<div>Mock H5 Plot</div>"

        with self.client.session_transaction() as sess:
            sess['output'] = 'Output with H5'
            sess['use_h5'] = True

        response = self.client.get('/results')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Output with H5', response.data)
        self.assertIn(b'Mock H5 Plot', response.data)

    def test_clear_old_plots(self):
        from app import clear_old_plots, PLOT_SAVE_DIR
        os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
        dummy_plot = os.path.join(PLOT_SAVE_DIR, 'dummy.png')
        with open(dummy_plot, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')

        self.assertTrue(os.path.exists(dummy_plot))
        clear_old_plots()
        self.assertFalse(os.path.exists(dummy_plot))

    def test_doe_missing_file(self):
        response = self.client.post('/doe', data={})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Error: Missing DOE spec file", response.data)

    def test_doe_valid_submission(self):
        doe_file_path = os.path.join(self.upload_folder, "doe_spec.json")
        with open(doe_file_path, 'w') as f:
            f.write('''{
                "version": "1.2",
                "spec": {
                    "design": "lhs",
                    "samples": 10
                }
            }''')

        with open(doe_file_path, 'rb') as doe_file:
            response = self.client.post('/doe', data={
                'doe_spec_file': (doe_file, 'doe_spec.json'),
                'out_dir_val': './results',
                'pref_val': 'TestDOE',
                'doe_algo': 'lhs',
                'log_time': 'yes',
                'additional_command': '-seed 999'
            }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers['Location'].endswith('/results'))

    def test_predict_missing_files(self):
        response = self.client.post('/predict', data={})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Error: Both model file and new data file are required", response.data)

    def test_predict_valid_submission(self):
        model_file_path = os.path.join(self.upload_folder, "model.h5")
        new_data_file_path = os.path.join(self.upload_folder, "new_data.csv")

        with open(model_file_path, 'w') as f:
            f.write("mock model content")
        with open(new_data_file_path, 'w') as f:
            f.write("mock,new,data")

        with open(model_file_path, 'rb') as model_file, open(new_data_file_path, 'rb') as new_data_file:
            response = self.client.post('/predict', data={
                'model_file': (model_file, 'model.h5'),
                'new_data_file': (new_data_file, 'new_data.csv'),
                'out_dir_val': './results',
                'pref_val': 'PredictRun',
                'log_time': 'yes',
                'plots': 'yes',
                'save_model': 't',
                'model_name': 'my_model',
                'seed_val': '123',
                'additional_command': '-seed 999'
            }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers['Location'].endswith('/results'))

if __name__ == '__main__':
    unittest.main()
