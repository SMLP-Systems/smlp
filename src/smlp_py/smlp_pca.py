# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
import json
#from sklearn.preprocessing import StandardScaler

class SmlpPCA:
    def __init__(self):
        self._pca_logger = None
        self.pca_model = None
        self._DEF_PCA_FEATURES_PRED = 0
        self.pca_params_dict = {
            'pca_feat_count_for_prediction': {'abbr':'pca_pred', 'default':self._DEF_PCA_FEATURES_PRED, 'type':int,
                'help':'Count of features selected by pca algorithm '  +
                    '[default: {}]'.format(str(self._DEF_PCA_FEATURES_PRED))},
        }
    def set_logger(self, logger):
        self._pca_logger = logger
        


    def create_pca_based_spec(self, X: pd.DataFrame, y: pd.DataFrame, original_spec_path: str):
        if self.pca_model is None:
            raise ValueError("PCA model has not been trained. Call smlp_pca() first.")

        self._pca_logger.info('Generating PCA-based spec file: start')

        # Read original spec file
        with open(original_spec_path, "r") as f:
            raw_text = f.read().replace('\t', '    ')
            spec_data = json.loads(raw_text)

        # Prepare output directory
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        output_dir = os.path.join(base_dir, "regr_smlp", "models")
        os.makedirs(output_dir, exist_ok=True)

        # Collect original knob ranges
        knob_ranges = {}
        original_variables = spec_data.get('variables', [])
        for item in original_variables:
            if item.get('interface') in ['knob', 'input']:
                knob_ranges[item['label']] = item['range']

        # PCA components and feature information
        components = self.pca_model.components_
        feature_means = self.pca_model.mean_
        feature_names = X.columns.tolist()
        pca_knobs = []
        pca_equations_log = ["PCA Component Equations:"]
        # Create PCA knobs with corrected ranges and save equations
        for i, component in enumerate(components):
            pc_name = f"PC{i+1}"

            min_pc, max_pc = 0, 0
            offset = 0.0
            terms = []

            for weight, feature in zip(component, feature_names):
                if feature not in knob_ranges:
                    continue
                feature_min, feature_max = knob_ranges[feature]
                if weight >= 0:
                    min_pc += weight * feature_min
                    max_pc += weight * feature_max
                else:
                    min_pc += weight * feature_max
                    max_pc += weight * feature_min

                # Accumulate the offset for mean centering
                feature_idx = feature_names.index(feature)

                offset -= weight * feature_means[feature_idx]

                # Build equation term
                if abs(weight) > 1e-6:
                    terms.append(f"{weight:.4f} * {feature}")

            # Final corrected min/max with offset
            estimated_min = min(min_pc, max_pc) + offset
            estimated_max = max(min_pc, max_pc) + offset
            estimated_range = [round(estimated_min, 4), round(estimated_max, 4)]

            # Create knob
            pca_knobs.append({
                "label": pc_name,
                "interface": "knob",
                "type": "real",
                "range": estimated_range,
                "rad-rel": 0.05
            })

            # Create readable equation for PCA component
            equation = " + ".join(terms)
            equation += f" + {offset:.4f}"
            pca_equations_log.append(f"{pc_name} ≈ {equation}")

        # Keep outputs unchanged
        output_variables = []
        for item in original_variables:
            if item.get('interface') == 'output':
                output_variables.append(item)

        # Combine into new spec
        new_variables = pca_knobs + output_variables
        new_spec = {
            "version": spec_data.get("version", "1.2"),
            "variables": new_variables,
            "objectives": spec_data.get("objectives", {})
        }

        # Save new PCA spec
        output_spec_path = os.path.join(output_dir, "pca_generated.spec")
        with open(output_spec_path, "w") as f:
            json.dump(new_spec, f, indent=4)

        self._pca_logger.info(f"PCA-based spec saved to {output_spec_path}")

        # Save PCA component equations
        pca_equations_path = os.path.join(output_dir, "pca_component_equations.txt")
        with open(pca_equations_path, "w") as f:
            f.write("\n".join(pca_equations_log))

        self._pca_logger.info(f"PCA component equations saved to {pca_equations_path}")
        self._pca_logger.info('Generating PCA-based spec file: end')

        return "OK"

    def smlp_pca(self, X: pd.DataFrame, y: pd.DataFrame, feat_count: int , spec_path: str):
        
        if X.shape[1] == 0:
            return X , None

        self._pca_logger.info('PCA feature compression: start')

        self.pca_model = PCA(n_components=feat_count)
        X_pca = self.pca_model.fit_transform(X)

        pca_columns = [f'PC{i+1}' for i in range(feat_count)]
        X_pca_df = pd.DataFrame(X_pca, index=X.index, columns=pca_columns)

        self._pca_logger.info(f'PCA applied: Reduced {X.shape[1]} features to {feat_count} components')
        self._pca_logger.info('PCA feature compression: end')

        # Go back two levels from smlp_py to project root
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        # Target output directory
        output_dir = os.path.join(base_dir, "regr_smlp", "models")
        os.makedirs(output_dir, exist_ok=True)

        # Save PCA components
        pca_components_path = os.path.join(output_dir, "pca_components.csv")
        X_pca_df.to_csv(pca_components_path, index=False)
        self._pca_logger.info(f"PCA components saved to {pca_components_path}")

        # Save combined PCA components and outputs
        X_pca_and_outputs = pd.concat([X_pca_df, y.loc[X_pca_df.index]], axis=1)
        pca_full_data_path = os.path.join(output_dir, "pca_full_data.csv")
        X_pca_and_outputs.to_csv(pca_full_data_path, index=False)
        self._pca_logger.info(f"PCA components with outputs saved to {pca_full_data_path}")

        return X_pca_df, self.pca_model

    def inverse_transform(self, X_pca , X):
        # Converts PCA features back to original feature space using stored PCA model.

        if self.pca_model is None:
            raise ValueError("PCA model has not been trained. Call smlp_pca() first.")

        X_reconstructed = pd.DataFrame(self.pca_model.inverse_transform(X_pca), index=X_pca.index)
        X_reconstructed.columns = list(X.columns)

        return X_reconstructed

    def get_feature_equations(self, X_pca, X):
        # Generate a dictionary mapping each original feature to its equation in terms of PCA-transformed components .

        if self.pca_model is None:
            raise ValueError("PCA model has not been trained. Call smlp_pca() first.")
        W = self.pca_model.components_  
        W_inv = np.linalg.pinv(W)  

        # Get the mean values of original features
        feature_means = self.pca_model.mean_
        feature_equations = {}
        equations_log = ["PCA Equations for Original Features:"]

        for i, feature_name in enumerate(X.columns):
            coefficients = W_inv[i, :]

            # Create equation for each feature
            feature_index = X.columns.get_loc(feature_name)
            equation = " + ".join(f"{coeff:.4f} * {pc}" for coeff, pc in zip(coefficients, X_pca.columns))
            equation += f" + {feature_means[feature_index]:.4f}"
            equation_terms = {pc: coeff for pc, coeff in zip(X_pca.columns, coefficients)}

            # Store each output equation
            feature_equations[feature_name] = {
                "terms": equation_terms,
                "offset": feature_means[i]
            }

            equations_log.append(f"{feature_name} ≈ {equation}")
        # Logs all equations, can be commented out if not relevant
        self._pca_logger.info("\n".join(equations_log))

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        # Target output directory
        output_dir = os.path.join(base_dir, "regr_smlp", "models")
        os.makedirs(output_dir, exist_ok=True)
        # Save PCA feature equations
        pca_equations_path = os.path.join(output_dir, "pca_equations.txt")
        with open(pca_equations_path, "w") as f:
            f.write("\n".join(equations_log))
        self._pca_logger.info(f"PCA feature equations saved to {pca_equations_path}")


        return feature_equations

