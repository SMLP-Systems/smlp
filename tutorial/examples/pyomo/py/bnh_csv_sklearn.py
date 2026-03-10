#!/usr/bin/python3.12
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from sys import argv
from math import inf
from hashlib import sha256

# ============================================================================
# SKLEARN SURROGATE MODEL IMPORTS
# ============================================================================
import pickle
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================================
# DEFINE CONSTRAINT FUNCTIONS (Python functions)
# ============================================================================

def constraint_C1(x1, x2):
    """
    C1: (x1 - 5)^2 + x2^2 ≤ 25
    Returns: constraint value (≤ 0 means satisfied)
    """
    return (x1 - 5)**2 + x2**2 - 25

def constraint_C2(x1, x2):
    """
    C2: (x1 - 8)^2 + (x2 + 3)^2 ≥ 7.7
    Converted to: -[(x1 - 8)^2 + (x2 + 3)^2] + 7.7 ≤ 0
    Returns: constraint value (≤ 0 means satisfied)
    """
    return -((x1 - 8)**2 + (x2 + 3)**2) + 7.7


# ============================================================================
# SKLEARN SURROGATE MODEL BUILDER
# ============================================================================

MODEL_PATH = "surrogate_F1_F2.pkl"


def build_and_save_sklearn_models(
    data: pd.DataFrame,
    rootpath: str = ".",
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict:
    """
    Train a single MultiOutputRegressor surrogate model for F1 and F2
    jointly, evaluate it, and persist it to disk via joblib.

    Architecture: StandardScaler → MultiOutputRegressor(GradientBoostingRegressor)

    Parameters
    ----------
    data        : DataFrame with columns X1, X2, F1, F2
    rootpath    : directory where the .pkl file will be written
    random_state: reproducibility seed
    test_size   : fraction of data held out for evaluation

    Returns
    -------
    dict with key 'model' (fitted Pipeline) and 'metrics' (evaluation dict).
    """
    print("\n" + "=" * 80)
    print("BUILDING SKLEARN MULTI-OUTPUT SURROGATE MODEL")
    print("=" * 80)

    X = data[["X1", "X2"]].values
    Y = data[["F1", "F2"]].values          # shape (n, 2)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    print(f"\nData split: {len(X_train)} train / {len(X_test)} test samples")
    print("Targets   : F1, F2  (predicted jointly)")

    # ------------------------------------------------------------------
    # Pipeline: StandardScaler + MultiOutputRegressor(GBR)
    # ------------------------------------------------------------------
    gbr_params = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=random_state,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mor",    MultiOutputRegressor(GradientBoostingRegressor(**gbr_params))),
    ])

    print("\nTraining multi-output surrogate model (F1 + F2) …")
    model.fit(X_train, Y_train)

    # ------------------------------------------------------------------
    # Evaluate on held-out test set — per output
    # ------------------------------------------------------------------
    Y_pred = model.predict(X_test)         # shape (n_test, 2)

    print("\n" + "-" * 80)
    print("MODEL EVALUATION (held-out test set)")
    print("-" * 80)

    metrics = {}
    for i, label in enumerate(["F1", "F2"]):
        r2   = r2_score(Y_test[:, i], Y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(Y_test[:, i], Y_pred[:, i]))
        mae  = mean_absolute_error(Y_test[:, i], Y_pred[:, i])
        print(f"\n  {label} — Test-set metrics:")
        print(f"    R²   : {r2:.6f}")
        print(f"    RMSE : {rmse:.6f}")
        print(f"    MAE  : {mae:.6f}")
        metrics[label] = {"r2": r2, "rmse": rmse, "mae": mae}

    # ------------------------------------------------------------------
    # 5-fold cross-validation R² (multi-output → mean across outputs)
    # ------------------------------------------------------------------
    print("\n  5-fold cross-validation R² (full dataset, mean across F1+F2):")
    cv_scores = cross_val_score(model, X, Y, cv=5, scoring="r2")
    print(f"    mean={cv_scores.mean():.4f}  std={cv_scores.std():.4f}")

    # ------------------------------------------------------------------
    # Persist the single model file
    # ------------------------------------------------------------------
    model_path = os.path.join(rootpath, MODEL_PATH)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n✓ Multi-output surrogate model saved → '{model_path}'")

    return {"model": model, "metrics": metrics}


def load_sklearn_models(rootpath: str = "."):
    """
    Load the previously saved multi-output surrogate model from disk.

    Returns the fitted Pipeline or None if the file is missing.
    """
    model_path = os.path.join(rootpath, MODEL_PATH)
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"\n✓ Loaded existing surrogate model from '{model_path}'")
        return model
    return None


# ============================================================================
# DEFINE OPTIMIZATION PROBLEM FOR PYMOO
# ============================================================================

class ParetoFromCSVData(Problem):
    """
    Multi-objective optimization using CSV data.

    Variables : X1, X2
    Objectives: F1(X1, X2), F2(X1, X2)  — predicted by sklearn surrogate
                models (GradientBoostingRegressor) with scipy interpolation
                as a fallback for out-of-hull points.
    Constraints: C1, C2

    NO ANALYTICAL FUNCTIONS FOR OBJECTIVES — PURE DATA-DRIVEN
    """

    def __init__(
        self,
        csv: str = "objectives_data.csv",
        rootpath: str = ".",
        retrain: bool = True,
    ):
        super().__init__(
            n_var=2,
            n_obj=2,
            n_constr=2,
            xl=np.array([0, 0]),
            xu=np.array([5, 3]),
        )
        self.data = pd.read_csv(csv)

        print(f"\nLoaded {len(self.data)} data points from CSV:")
        print(self.data.head(10))

        # ------------------------------------------------------------------
        # Scipy interpolators (kept as fallback / reference)
        # ------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("CREATING SCIPY INTERPOLATORS FROM CSV DATA")
        print("=" * 80)

        self.points    = self.data[["X1", "X2"]].values
        self.f1_values = self.data["F1"].values
        self.f2_values = self.data["F2"].values

        self.f1_interpolator_linear  = LinearNDInterpolator(self.points, self.f1_values)
        self.f1_interpolator_nearest = NearestNDInterpolator(self.points, self.f1_values)
        self.f2_interpolator_linear  = LinearNDInterpolator(self.points, self.f2_values)
        self.f2_interpolator_nearest = NearestNDInterpolator(self.points, self.f2_values)

        # ------------------------------------------------------------------
        # Sklearn surrogate model (single multi-output model)
        # ------------------------------------------------------------------
        model = None if retrain else load_sklearn_models(rootpath)

        if model is None:
            result = build_and_save_sklearn_models(self.data, rootpath)
            model = result["model"]

        self.surrogate = model  # Pipeline: scaler → MultiOutputRegressor(GBR)

    # ------------------------------------------------------------------
    # Objective evaluation helpers
    # ------------------------------------------------------------------

    def _sklearn_predict(self, x1, x2):
        """
        Predict both F1 and F2 in one call via the multi-output surrogate.
        Returns array of shape (n, 2): column 0 = F1, column 1 = F2.
        """
        X = np.column_stack([x1, x2])
        return self.surrogate.predict(X).astype(float)  # (n, 2)

    def get_F1_from_data(self, x1, x2):
        """
        F1 from multi-output surrogate (column 0).
        Fallback: scipy NearestND for any NaN values.
        """
        result = self._sklearn_predict(x1, x2)[:, 0]
        if np.isnan(result).any():
            nan_mask = np.isnan(result)
            result[nan_mask] = self.f1_interpolator_nearest(x1[nan_mask], x2[nan_mask])
        return result

    def get_F2_from_data(self, x1, x2):
        """
        F2 from multi-output surrogate (column 1).
        Fallback: scipy NearestND for any NaN values.
        """
        result = self._sklearn_predict(x1, x2)[:, 1]
        if np.isnan(result).any():
            nan_mask = np.isnan(result)
            result[nan_mask] = self.f2_interpolator_nearest(x1[nan_mask], x2[nan_mask])
        return result

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate objectives and constraints.

        X: 2D array where each row is [x1, x2]

        IMPORTANT: F1 and F2 are obtained via sklearn surrogate models
        trained on the CSV data (no hardcoded analytical functions).
        """
        x1 = X[:, 0]
        x2 = X[:, 1]

        f1 = self.get_F1_from_data(x1, x2)
        f2 = self.get_F2_from_data(x1, x2)

        g1 = constraint_C1(x1, x2)
        g2 = constraint_C2(x1, x2)

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])


# ============================================================================
# MAIN
# ============================================================================

def main(rootpath: str = ".", timeout: float = 5000, retrain: bool = False) -> str:
    print("=" * 80)
    print("PARETO FRONT FROM CSV DATA (NO ANALYTICAL FUNCTIONS)")
    print("=" * 80)
    print("\nIMPORTANT: F1 and F2 are predicted by sklearn surrogate models")
    print("trained exclusively on the CSV data — no analytical functions used!")

    # ------------------------------------------------------------------
    # Create problem (trains / loads sklearn models internally)
    # ------------------------------------------------------------------
    problem = ParetoFromCSVData(
        csv=rootpath + "/objectives_data.csv",
        rootpath=rootpath,
        retrain=retrain,
    )

    # ------------------------------------------------------------------
    # NSGA-II optimisation
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RUNNING NSGA-II OPTIMIZATION")
    print("=" * 80)

    algorithm = NSGA2(
        pop_size=100,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", 200)

    print("\nOptimization Settings:")
    print("  Objectives : F1, F2 via sklearn GBR surrogate (+ scipy fallback)")
    print("  Variables  : X1 ∈ [0, 5], X2 ∈ [0, 3]")
    print("  Constraints:")
    print("    C1: (x1-5)² + x2² ≤ 25")
    print("    C2: (x1-8)² + (x2+3)² ≥ 7.7")
    print("  Population : 100")
    print("  Generations: 200")
    print()

    res = minimize(problem, algorithm, termination, seed=42, verbose=True)

    # ------------------------------------------------------------------
    # Extract and report Pareto front
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PARETO FRONT RESULTS")
    print("=" * 80)

    pareto_X = res.X
    pareto_F = res.F

    print(f"\nNumber of Pareto optimal solutions: {len(pareto_X)}")

    pareto_df = pd.DataFrame({
        "Solution": range(1, len(pareto_X) + 1),
        "X1": pareto_X[:, 0],
        "X2": pareto_X[:, 1],
        "F1": pareto_F[:, 0],
        "F2": pareto_F[:, 1],
    })

    pareto_df["C1_value"]    = pareto_df.apply(lambda r: constraint_C1(r["X1"], r["X2"]), axis=1)
    pareto_df["C2_value"]    = pareto_df.apply(lambda r: constraint_C2(r["X1"], r["X2"]), axis=1)
    pareto_df["C1_satisfied"] = pareto_df["C1_value"] <= 1e-6
    pareto_df["C2_satisfied"] = pareto_df["C2_value"] <= 1e-6
    pareto_df["All_constraints_OK"] = pareto_df["C1_satisfied"] & pareto_df["C2_satisfied"]

    print("\n" + "-" * 80)
    print("Pareto Optimal Solutions (First 15):")
    print("-" * 80)
    display_cols = ["Solution", "X1", "X2", "F1", "F2", "All_constraints_OK"]
    print(pareto_df[display_cols].head(15).to_string(index=False))

    print("\n" + "-" * 80)
    print("Summary Statistics:")
    print("-" * 80)
    print(f"X1 range: [{pareto_X[:, 0].min():.3f}, {pareto_X[:, 0].max():.3f}]")
    print(f"X2 range: [{pareto_X[:, 1].min():.3f}, {pareto_X[:, 1].max():.3f}]")
    print(f"F1 range: [{pareto_F[:, 0].min():.3f}, {pareto_F[:, 0].max():.3f}]")
    print(f"F2 range: [{pareto_F[:, 1].min():.3f}, {pareto_F[:, 1].max():.3f}]")
    print(f"Feasible solutions: {pareto_df['All_constraints_OK'].sum()} / {len(pareto_df)}")

    pareto_df.to_csv(rootpath + "/pareto_front_results_sklearn_tab.csv", index=False)
    print(f"\n✓ Pareto front saved to 'pareto_front_results_sklearn_tab.csv'")

    # ------------------------------------------------------------------
    # Analyse original CSV data
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANALYZING ORIGINAL CSV DATA")
    print("=" * 80)

    data = problem.data
    data["C1_value"]     = data.apply(lambda r: constraint_C1(r["X1"], r["X2"]), axis=1)
    data["C2_value"]     = data.apply(lambda r: constraint_C2(r["X1"], r["X2"]), axis=1)
    data["C1_satisfied"] = data["C1_value"] <= 1e-6
    data["C2_satisfied"] = data["C2_value"] <= 1e-6
    data["Feasible"]     = data["C1_satisfied"] & data["C2_satisfied"]

    print(f"\nOriginal CSV data points:")
    print(f"  Total points     : {len(data)}")
    print(f"  Feasible points  : {data['Feasible'].sum()}")
    print(f"  Infeasible points: {(~data['Feasible']).sum()}")

    # ------------------------------------------------------------------
    # Visualisation — 2 plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    feasible_csv   = data[data["Feasible"]]
    infeasible_csv = data[~data["Feasible"]]

    # Plot 1: Objective space
    ax1.scatter(infeasible_csv["F1"], infeasible_csv["F2"], c="lightcoral", s=40, alpha=0.4, marker="x", label="CSV Infeasible")
    ax1.scatter(feasible_csv["F1"],   feasible_csv["F2"],   c="lightblue",  s=40, alpha=0.5, marker="o", label="CSV Feasible")
    ax1.scatter(pareto_F[:, 0], pareto_F[:, 1], c="darkgreen", s=100, alpha=0.9,
                edgecolors="black", linewidths=2, marker="D", label="Optimized Pareto Front", zorder=10)

    sorted_indices = np.argsort(pareto_F[:, 0])
    ax1.plot(pareto_F[sorted_indices, 0], pareto_F[sorted_indices, 1], "g--", linewidth=2, alpha=0.5, zorder=5)

    ax1.set_xlabel("F1", fontsize=13, fontweight="bold")
    ax1.set_ylabel("F2", fontsize=13, fontweight="bold")
    ax1.set_title("Pareto Front in Objective Space\n(sklearn GBR surrogate)", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11, loc="best")
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Plot 2: Variable / decision space
    x1_grid = np.linspace(0, 5, 100)
    x2_grid = np.linspace(0, 3, 100)
    X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)

    C1_mesh = (X1_mesh - 5)**2 + X2_mesh**2
    ax2.contour(X1_mesh, X2_mesh, C1_mesh, levels=[25],   colors="red",  linewidths=2.5, linestyles="--")
    ax2.contourf(X1_mesh, X2_mesh, C1_mesh, levels=[0, 25], colors=["lightgreen"], alpha=0.15)

    C2_mesh = (X1_mesh - 8)**2 + (X2_mesh + 3)**2
    ax2.contour(X1_mesh, X2_mesh, C2_mesh, levels=[7.7],      colors="blue", linewidths=2.5, linestyles="--")
    ax2.contourf(X1_mesh, X2_mesh, C2_mesh, levels=[7.7, 100], colors=["lightblue"], alpha=0.15)

    ax2.scatter(infeasible_csv["X1"], infeasible_csv["X2"], c="lightcoral", s=35, marker="x", alpha=0.5, label="CSV Infeasible")
    ax2.scatter(feasible_csv["X1"],   feasible_csv["X2"],   c="lightblue",  s=35, marker="o", alpha=0.5, label="CSV Feasible")
    ax2.scatter(pareto_X[:, 0], pareto_X[:, 1], c="darkgreen", s=100, alpha=0.9,
                edgecolors="black", linewidths=2, marker="D", label="Pareto Solutions", zorder=10)
    ax2.plot(pareto_X[sorted_indices, 0], pareto_X[sorted_indices, 1], "g--", linewidth=2, alpha=0.5, zorder=5)

    ax2.set_xlabel("X1", fontsize=13, fontweight="bold")
    ax2.set_ylabel("X2", fontsize=13, fontweight="bold")
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 3)
    ax2.set_title("Pareto Front in Variable Space\n(Feasible Region = Green ∩ Blue)", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11, loc="best")
    ax2.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(rootpath + "/pareto_front_two_plots_sklearn.png", dpi=300, bbox_inches="tight")
    print("✓ Visualization saved as 'pareto_front_two_plots_sklearn.png'")

    if not inf == timeout:
        timer = fig.canvas.new_timer(interval=timeout, callbacks=[(plt.close, [], {})])
        timer.start()
    plt.show()

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. pareto_front_results_sklearn_tab.csv  — Pareto optimal solutions")
    print("  2. pareto_front_two_plots_sklearn.png    — Objective and variable space plots")
    print("  3. surrogate_F1_F2.pkl           — sklearn multi-output surrogate (F1 + F2)")
    print("\n✓ F1 and F2 predicted jointly by a single MultiOutputRegressor(GBR) surrogate")
    print("✓ Model trained solely on CSV data and saved for reuse")
    print("✓ Scipy NearestND interpolator used as NaN fallback only")
    print("=" * 80)

    return sha256(pareto_df.to_string().encode()).hexdigest()


if __name__ == "__main__":
    rootpath = "."         if len(argv) < 2 else argv[1]
    timeout  = inf         if len(argv) < 3 else float(argv[2])
    retrain  = False       if len(argv) < 4 else argv[3].lower() in ("1", "true", "yes")
    print(main(rootpath, timeout, retrain))
