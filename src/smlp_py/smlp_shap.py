import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

class SmlpShap:
    def __init__(self):
        self._shap_logger = None
        self._SHAP_FEATURES_PRED = 15
        self._DEF_ESTIMATOR_NUMBER = 100

    def set_logger(self, logger):
        self._shap_logger = logger
    
    def smlp_shap(self, X: pd.DataFrame, y: pd.Series, feat_cnt: int):
        if feat_cnt == 0 or X.shape[1] == 0:
            return [], np.zeros(X.shape[1])
        self._shap_logger.info(f'SHAP feature selection for response {y.name}: start')


        # Code added in order to reduce training time
        X_sample = X.sample(n=min(1000, len(X)), random_state=42)
        y_sample = y.loc[X_sample.index]

        # Train the model
        model = RandomForestRegressor(n_estimators=self._DEF_ESTIMATOR_NUMBER, n_jobs=-1)
        model.fit(X_sample, y_sample)

        r2 = r2_score(y_sample, model.predict(X_sample))
        if r2 < 0.01:
            self._shap_logger.warning(f"Model R² score for response {y.name} is very low (R² = {r2:.5f}); SHAP results may be uninformative.")
            return [], np.zeros(X_sample.shape[1])


        # Option 1: KernelExplainer (model independent, very slow)
        # This is extremely slow for large datasets.
        # explainer = shap.KernelExplainer(model.predict, X_sample)

        # Option 2: API Explainer (automatically selects an explainer type, slow for tree models)
        # explainer = shap.Explainer(model.predict, X_sample, feature_perturbation="interventional")

        # Option 3 (recommended): TreeExplainer (fast and gives similar results to Kernel and API explainers in testing)
        explainer = shap.TreeExplainer(model)
        
        # Compute mean absolute SHAP value per feature
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_importance = np.abs(np.array(shap_values)).mean(axis=(0, 1))
        else:
            shap_importance = np.abs(shap_values).mean(axis=0)
        
        if np.all(shap_importance == 0):
            self._shap_logger.warning(f"All SHAP values are 0 for response {y.name}; skipping SHAP feature selection.")
            return [], shap_importance
        
        # Rank features by SHAP importance in descending order
        feature_importance = pd.DataFrame({
            'Feature': X_sample.columns,
            'SHAP_Importance': shap_importance
        }).sort_values(by='SHAP_Importance', ascending=False)

        selected_features = feature_importance['Feature'].iloc[:feat_cnt].tolist()
        
        self._shap_logger.info(f'SHAP selected feature scores for response {y.name}:{feature_importance}')
        self._shap_logger.info(f'SHAP feature selection for response {y.name}: end')

        return selected_features , shap_importance