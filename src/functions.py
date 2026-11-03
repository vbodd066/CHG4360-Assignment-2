# =======================================================================
# FUNCTIONS
# Define reusable helper functions in this file.
# =======================================================================
import os

import matplotlib
matplotlib.use("Agg")                                   # write figures to file without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.constants import *


# -----------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------

# load the csv into a pandas df
def load_data():
    df = pd.read_csv("datasets/dataset_cell_growth_4factor.csv")
    return df


# hold out a test set that is only touched once, at the very end
def split_data(df):
    x = df[FEATURES]
    y = df[TARGET]
    return train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


# -----------------------------------------------------------------------
#                             MODELS
# -----------------------------------------------------------------------

# polynomial regression
def build_polynomial_pipeline():
    pipeline = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(include_bias=False),
        LinearRegression(),
    )
    param_grid = {"polynomialfeatures__degree": POLY_DEGREES}
    return pipeline, param_grid


# k-nearest neighbours
def build_knn_pipeline():
    pipeline = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(),
    )
    param_grid = {
        "kneighborsregressor__n_neighbors": KNN_NEIGHBORS,
        "kneighborsregressor__weights": KNN_WEIGHTS,
    }
    return pipeline, param_grid


# every combination in the grid is scored on the same folds so the two model families are compared fairly
def tune_model(pipeline, param_grid, x_train, y_train):
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=CV_SPLITS,
        scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
        refit="rmse",                                   # the best setting is refit on the full training set
        return_train_score=True,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    return search


# tidy version of GridSearchCV.cv_results_ with positive RMSE values, sorted best first
def cv_table(search):
    results = pd.DataFrame(search.cv_results_)
    param_cols = [c for c in results.columns if c.startswith("param_")]
    table = results[param_cols].copy()
    table.columns = [c.removeprefix("param_").split("__")[-1] for c in param_cols]
    table["train_rmse"] = -results["mean_train_rmse"] + 0.0       # + 0.0 turns -0.0 into 0.0
    table["cv_rmse"] = -results["mean_test_rmse"]
    table["cv_rmse_std"] = results["std_test_rmse"]
    table["train_r2"] = results["mean_train_r2"]
    table["cv_r2"] = results["mean_test_r2"]
    return table.sort_values("cv_rmse").reset_index(drop=True)


# -----------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------

def regression_metrics(y_true, y_pred, prefix):
    return {
        f"{prefix}_r2": r2_score(y_true, y_pred),
        f"{prefix}_rmse": root_mean_squared_error(y_true, y_pred),
    }


# score both tuned models once on the held-out test set
def evaluate_on_test(searches, x_train, x_test, y_train, y_test):
    rows = []
    for name, search in searches.items():
        model = search.best_estimator_
        y_pred = model.predict(x_test)
        rows.append({
            "model": name,
            "best_params": str({k.split("__")[-1]: v for k, v in search.best_params_.items()}),
            **regression_metrics(y_train, model.predict(x_train), "train"),
            **regression_metrics(y_test, y_pred, "test"),
        })
    table = pd.DataFrame(rows)
    save_table(table.round(4), "model_evaluation.csv")
    return table


# re-tune both models on growing random subsets of the training set and score them on the same test set
def learning_curve(x_train, y_train, x_test, y_test):
    rng = np.random.default_rng(seed=RANDOM_STATE)
    sizes = [n for n in LEARNING_CURVE_SIZES if n < len(x_train)] + [len(x_train)]
    # nested subsets: each larger subset contains the smaller ones, so the curve reflects adding data, not a new draw
    order = rng.permutation(len(x_train))
    rows = []
    for n_samples in sizes:
        x_sub, y_sub = x_train.iloc[order[:n_samples]], y_train.iloc[order[:n_samples]]

        poly_pipeline, poly_grid = build_polynomial_pipeline()
        knn_pipeline, knn_grid = build_knn_pipeline()
        # k cannot be larger than the number of samples in the smallest cross-validation training fold
        max_k = n_samples - int(np.ceil(n_samples / CV_SPLITS))
        knn_grid["kneighborsregressor__n_neighbors"] = [k for k in KNN_NEIGHBORS if k <= max_k]

        for name, pipeline, grid in [("Polynomial", poly_pipeline, poly_grid), ("KNN", knn_pipeline, knn_grid)]:
            search = tune_model(pipeline, grid, x_sub, y_sub)
            model = search.best_estimator_
            rows.append({
                "model": name,
                "training_samples": n_samples,
                "best_params": str({k.split("__")[-1]: v for k, v in search.best_params_.items()}),
                "train_rmse": root_mean_squared_error(y_sub, model.predict(x_sub)),
                "test_rmse": root_mean_squared_error(y_test, model.predict(x_test)),
            })
    table = pd.DataFrame(rows)
    save_table(table.round(4), "learning_curve.csv")
    return table


# -----------------------------------------------------------------------
# INTERPRETATION
# -----------------------------------------------------------------------

# vary one input at a time, holding the others at the reference condition
def sweep_feature(feature, x_all, n_points=SWEEP_POINTS, margin=0.15):
    low, high = x_all[feature].min(), x_all[feature].max()
    pad = (high - low) * margin
    values = np.linspace(low - pad, high + pad, n_points)
    if feature in ["Glucose [mM]", "Lactate [mM]"]:
        values = values[values >= 0]                       # concentrations cannot be negative
    sweep = pd.DataFrame({f: np.full(len(values), REFERENCE_CONDITION[f]) for f in FEATURES})
    sweep[feature] = values
    return sweep


# summary of each one-variable sweep inside the observed range, for both tuned models
def feature_effects(poly_model, knn_model, x_all):
    rows = []
    for feature in FEATURES:
        sweep = sweep_feature(feature, x_all, margin=0.0)
        for name, model in [("Polynomial", poly_model), ("KNN", knn_model)]:
            pred = np.clip(model.predict(sweep), 0, None)       # a negative growth rate is not physical
            best = int(np.argmax(pred))
            rows.append({
                "feature": feature,
                "model": name,
                "growth_at_min": pred[0],
                "growth_at_max": pred[-1],
                "value_at_peak_growth": sweep[feature].iloc[best],
                "peak_growth": pred[best],
                "growth_range": pred.max() - pred.min(),
            })
    table = pd.DataFrame(rows)
    save_table(table.round(3), "feature_effects.csv")
    return table


# -----------------------------------------------------------------------
# FIGURES
# -----------------------------------------------------------------------

def plot_polynomial_validation(poly_table):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    curve = poly_table.sort_values("degree")
    ax.plot(curve["degree"], curve["train_rmse"], "o--", color="tab:blue", label="Training")
    ax.errorbar(curve["degree"], curve["cv_rmse"], yerr=curve["cv_rmse_std"], fmt="o-", color="tab:orange",
                capsize=3, label="Cross-validation (+/- 1 SD)")
    ax.set_yscale("log")
    ax.set_xticks(POLY_DEGREES)
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("RMSE [d$^{-1}$] (log scale)")
    ax.set_title("Polynomial regression: training vs cross-validation error")
    ax.legend(fontsize=8)
    save_figure(fig, "polynomial_validation_curve.png")


def plot_knn_validation(knn_table, best_k):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for weights, color in zip(KNN_WEIGHTS, ["tab:green", "tab:purple"]):
        curve = knn_table[knn_table["weights"] == weights].sort_values("n_neighbors")
        ax.plot(curve["n_neighbors"], curve["cv_rmse"], "-", color=color, label=f"{weights} weights - CV")
        ax.plot(curve["n_neighbors"], curve["train_rmse"], "--", color=color, label=f"{weights} weights - train")
    ax.axvline(best_k, color="grey", linestyle=":", label=f"selected k = {best_k}")
    ax.set_xlabel("Number of neighbours k")
    ax.set_ylabel("RMSE [d$^{-1}$]")
    ax.set_title("KNN regression: training vs cross-validation error")
    ax.legend(fontsize=8)
    save_figure(fig, "knn_validation_curve.png")


def plot_learning_curve(lc_table):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, color in [("Polynomial", "tab:orange"), ("KNN", "tab:green")]:
        curve = lc_table[lc_table["model"] == name]
        ax.plot(curve["training_samples"], curve["test_rmse"], "o-", color=color, label=f"{name} - test")
    # KNN training error is left out: distance weighting reproduces every training sample, so it is almost always 0
    poly = lc_table[lc_table["model"] == "Polynomial"]
    ax.plot(poly["training_samples"], poly["train_rmse"], "o--", color="tab:orange", label="Polynomial - train")
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("RMSE [d$^{-1}$]")
    ax.set_title("Learning curve: error vs amount of training data")
    ax.legend(fontsize=8)
    save_figure(fig, "learning_curve.png")


def plot_feature_responses(poly_model, knn_model, x_all, y_all):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, feature in zip(axes.ravel(), FEATURES):
        sweep = sweep_feature(feature, x_all)
        ax.scatter(x_all[feature], y_all, s=8, color="lightgrey", label="Measured samples (all conditions)")
        ax.plot(sweep[feature], poly_model.predict(sweep), color="tab:orange", linewidth=2, label="Polynomial")
        ax.plot(sweep[feature], knn_model.predict(sweep), color="tab:green", linewidth=1.5, label="KNN")
        # shade the region outside the observed range, where both models are extrapolating
        low, high = x_all[feature].min(), x_all[feature].max()
        ax.axvspan(sweep[feature].min(), low, color="red", alpha=0.08)
        ax.axvspan(high, sweep[feature].max(), color="red", alpha=0.08, label="Outside observed range")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlabel(feature)
        ax.set_ylim(-0.25, 0.85)
    axes[0, 0].set_ylabel("Specific growth rate [d$^{-1}$]")
    axes[1, 0].set_ylabel("Specific growth rate [d$^{-1}$]")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    reference = ", ".join(f"{k} = {v:g}" for k, v in REFERENCE_CONDITION.items())
    fig.suptitle(f"One-factor-at-a-time response (other inputs held at {reference})", fontsize=10)
    save_figure(fig, "feature_responses.png")


# -----------------------------------------------------------------------
# EXPORT HELPERS
# -----------------------------------------------------------------------

def save_table(table, filename, index=False):
    os.makedirs("tables", exist_ok=True)
    table.to_csv(os.path.join("tables", filename), index=index)


def save_figure(fig, filename):
    os.makedirs("figures", exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", filename), dpi=150)
    plt.close(fig)
