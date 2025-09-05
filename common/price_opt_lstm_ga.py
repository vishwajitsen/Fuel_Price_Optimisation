#!/usr/bin/env python3
"""
price_opt_stack_xgb_lgbm_ga.py

- Loads historical data and today JSON (same format you used earlier).
- Builds features (calendar, competitor, lags, rolling windows, margins).
- Selects top features.
- Trains a Stacked ensemble: LightGBM (tuned via Optuna) + XGBoost (tuned via Optuna) + RandomForest,
  with a Ridge meta-learner. Falls back to XGBoost+RF if LightGBM not installed.
- Wraps predictor for use inside the existing GeneticOptimizer (keeps GA unchanged).
- Saves outputs and plots.

Requirements:
  pip install pandas numpy scikit-learn matplotlib optuna xgboost lightgbm joblib
(If lightgbm isn't available, the script still runs with xgboost+rf.)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import optuna
import joblib

# XGBoost
try:
    import xgboost as xgb  # type: ignore
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# LightGBM optional
try:
    import lightgbm as lgb  # type: ignore
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except Exception:
    HAS_LGB = False

# ---------------------------
# Utilities
# ---------------------------

def ensure_dir(p: str):
    if p:
        Path(p).mkdir(parents=True, exist_ok=True)

def save_json(obj: Dict[str, Any], out_file: str):
    ensure_dir(os.path.dirname(out_file))
    Path(out_file).write_text(json.dumps(obj, indent=2))
    print(f"Saved JSON: {out_file}")

def save_excel(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_excel(path, index=False)
    print(f"Saved Excel: {path}")

def save_plot(fig, path: str):
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")

# ---------------------------
# Data loading & validation
# ---------------------------

def load_history(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"History file not found: {path}")
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("History file must contain a 'date' column")
    df["date"] = pd.to_datetime(df["date"])
    expected = {"date","price","cost","comp1_price","comp2_price","comp3_price","volume"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"History file is missing columns: {missing}")
    df = df.sort_values("date").reset_index(drop=True)
    return df

def load_today(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Today JSON not found: {path}")
    j = json.loads(p.read_text())
    required = {"date","price","cost","comp1_price","comp2_price","comp3_price"}
    missing = required - set(j.keys())
    if missing:
        raise ValueError(f"today json missing keys: {missing}")
    today = {
        "date": pd.to_datetime(j["date"]),
        "price": float(j["price"]),
        "cost": float(j["cost"]),
        "comp1_price": float(j["comp1_price"]),
        "comp2_price": float(j["comp2_price"]),
        "comp3_price": float(j["comp3_price"]),
    }
    today["comp_mean"] = float(np.mean([today["comp1_price"], today["comp2_price"], today["comp3_price"]]))
    today["comp_min"] = float(min(today["comp1_price"], today["comp2_price"], today["comp3_price"]))
    today["comp_max"] = float(max(today["comp1_price"], today["comp2_price"], today["comp3_price"]))
    return today

# ---------------------------
# EDA
# ---------------------------

def data_summary(df: pd.DataFrame):
    print("\n--- DATA SUMMARY ---")
    print("Shape:", df.shape)
    print("Date range:", df["date"].min().date(), "to", df["date"].max().date())
    print("\nSample:")
    print(df.head(5).to_string(index=False))
    print("\nMissing values per column:")
    print(df.isna().sum())
    print("\nNumeric summary:")
    print(df.describe().transpose())
    print("--- END ---\n")

# ---------------------------
# Feature engineering
# ---------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear
    df["trend"] = np.arange(len(df))
    return df

def add_competitor_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    comps = df[["comp1_price","comp2_price","comp3_price"]].astype(float)
    df["comp_mean"] = comps.mean(axis=1)
    df["comp_min"] = comps.min(axis=1)
    df["comp_max"] = comps.max(axis=1)
    df["comp_spread"] = df["comp_max"] - df["comp_min"]
    df["price_gap_mean"] = df["price"] - df["comp_mean"]
    df["margin"] = df["price"] - df["cost"]
    return df

def add_lag_features(df: pd.DataFrame, lags: List[int] = [1,7,14]) -> pd.DataFrame:
    df = df.copy().sort_values("date")
    for lag in lags:
        df[f"volume_lag{lag}"] = df["volume"].shift(lag)
        df[f"price_lag{lag}"] = df["price"].shift(lag)
    # rolling windows
    df["volume_roll7"] = df["volume"].shift(1).rolling(7).mean()
    df["price_roll7"] = df["price"].shift(1).rolling(7).mean()
    df["price_gap_mean_lag1"] = df["price_gap_mean"].shift(1)
    df["price_gap_mean_roll7"] = df["price_gap_mean"].shift(1).rolling(7).mean()
    df = df.dropna().reset_index(drop=True)
    return df

def build_features(df: pd.DataFrame, with_lags: bool = True) -> pd.DataFrame:
    df2 = add_calendar_features(df)
    df2 = add_competitor_features(df2)
    if with_lags:
        df2 = add_lag_features(df2)
    return df2

# ---------------------------
# Train/test split & feature selection
# ---------------------------

def time_train_test_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n < 20:
        raise ValueError("Not enough data for reliable split")
    split_at = int(n * (1.0 - test_size))
    train = df.iloc[:split_at].reset_index(drop=True)
    test = df.iloc[split_at:].reset_index(drop=True)
    return train, test

def select_top_features_by_importance(X_train: pd.DataFrame, y_train: pd.Series, top_k: int = 15) -> List[str]:
    # Quick tree-based importance
    model = GradientBoostingRegressor(random_state=42, n_estimators=150)
    model.fit(X_train.fillna(0), y_train)
    feat_imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nFeature importances (top):")
    print(feat_imp.head(top_k).to_string())
    return list(feat_imp.head(top_k).index)

# ---------------------------
# Model training: Optuna tuning helpers
# ---------------------------

def objective_lgb(trial, X, y, tss_splits=3):
    if not HAS_LGB:
        return float("inf")
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42
    }
    tss = TimeSeriesSplit(n_splits=tss_splits)
    rmses = []
    for tr_idx, val_idx in tss.split(X):
        Xtr, Xval = X.iloc[tr_idx], X.iloc[val_idx]
        ytr, yval = y.iloc[tr_idx], y.iloc[val_idx]
        model = LGBMRegressor(**param)
        model.fit(Xtr, ytr)
        preds = model.predict(Xval)
        rmses.append(np.sqrt(mean_squared_error(yval, preds)))
    return float(np.mean(rmses))

def objective_xgb(trial, X, y, tss_splits=3):
    if not HAS_XGB:
        return float("inf")
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "random_state": 42,
        "objective": "reg:squarederror"
    }
    tss = TimeSeriesSplit(n_splits=tss_splits)
    rmses = []
    for tr_idx, val_idx in tss.split(X):
        Xtr, Xval = X.iloc[tr_idx], X.iloc[val_idx]
        ytr, yval = y.iloc[tr_idx], y.iloc[val_idx]
        model = XGBRegressor(**param)
        model.fit(Xtr, ytr)
        preds = model.predict(Xval)
        rmses.append(np.sqrt(mean_squared_error(yval, preds)))
    return float(np.mean(rmses))

# ---------------------------
# Wrapper predictor for GA compatibility
# ---------------------------

class EnsemblePredictor:
    """
    Wraps a stacked sklearn Pipeline / StackingRegressor.
    Exposes .predict(X_df) which accepts a DataFrame of feature columns (selected_features order).
    """
    def __init__(self, model, scaler: StandardScaler, selected_features: List[str], df_history: pd.DataFrame):
        self.model = model
        self.scaler = scaler
        self.selected_features = selected_features
        self.df_history = df_history

    def predict(self, X):
        # Accept DataFrame or numpy array; return 1D numpy array
        if isinstance(X, pd.DataFrame):
            X_in = X[self.selected_features].copy()
        else:
            X_in = pd.DataFrame(X, columns=self.selected_features)
        # impute missing
        X_in = X_in.fillna(0.0)
        # scale
        X_scaled = self.scaler.transform(X_in.values)
        preds = self.model.predict(X_scaled)
        return np.asarray(preds).reshape(-1)

# ---------------------------
# GeneticOptimizer (same as your existing implementation)
# ---------------------------

class GeneticOptimizer:
    def __init__(
        self,
        predictor,                  # trained demand model with .predict(X_df) or .predict(array)
        df_history: pd.DataFrame,
        today: Dict[str, Any],
        selected_features: List[str],
        avg_cv_r2: float = 0.0,     # average cross-validated R2 of surrogate model
        r2_alpha: float = 0.7,      # baseline weight
        r2_beta: float = 0.3,       # multiplier on avg_cv_r2
        pop_size: int = 100,
        generations: int = 50,
        crossover_prob: float = 0.6,
        mutation_prob: float = 0.3,
        elite_size: int = 5,
        price_step: float = 0.01,
        seed: int = 42
    ):
        self.predictor = predictor
        self.df_history = df_history
        self.today = today
        self.selected_features = selected_features
        self.avg_cv_r2 = float(max(0.0, avg_cv_r2))
        self.r2_alpha = float(r2_alpha)
        self.r2_beta = float(r2_beta)
        self.pop_size = pop_size
        self.generations = generations
        self.cx_prob = crossover_prob
        self.mut_prob = mutation_prob
        self.elite_size = elite_size
        self.price_step = price_step
        self.rng = np.random.default_rng(seed)

        # compute price bounds for GA search
        comp_mean = today["comp_mean"]
        low = max(today["cost"] + 0.01, comp_mean - 2.0)
        high = comp_mean + 2.0
        if high - low < 0.5:
            low = comp_mean - 0.5
            high = comp_mean + 0.5
        self.low = round(low - 0.5, 2)
        self.high = round(high + 0.5, 2)

        # guardrail params
        self.max_daily_change = 1.0
        self.min_margin = 0.10
        self.keep_within_competitors = True
        self.comp_band = 0.75

    def apply_guardrails(self, p: float) -> float:
        p = float(p)
        p = max(p, self.today["cost"] + self.min_margin)
        p = min(p, self.today["price"] + self.max_daily_change)
        p = max(p, self.today["price"] - self.max_daily_change)
        if self.keep_within_competitors:
            p = min(p, self.today["comp_mean"] + self.comp_band)
            p = max(p, self.today["comp_mean"] - self.comp_band)
        return round(p, 2)

    def _features_for_prices(self, prices: np.ndarray) -> pd.DataFrame:
        base = build_today_base_features(self.df_history, self.today)
        rows = []
        for p in prices:
            feat = base.copy()
            feat["price"] = float(p)
            feat["margin"] = float(p - base["cost"])
            feat["price_gap_mean"] = float(p - base["comp_mean"])
            rows.append(feat)
        X = pd.DataFrame(rows)
        for c in self.selected_features:
            if c not in X.columns:
                X[c] = 0.0
        X = X[self.selected_features]
        return X

    def fitness(self, price: float) -> float:
        p_guarded = self.apply_guardrails(price)
        X = self._features_for_prices(np.array([p_guarded]))
        try:
            v = float(self.predictor.predict(X)[0])
        except Exception:
            v = float(self.predictor.predict(X.values)[0])
        v = max(0.0, v)
        profit = (p_guarded - self.today["cost"]) * v
        gen_weight = (self.r2_alpha + self.r2_beta * max(0.0, self.avg_cv_r2))
        fitness_value = profit * gen_weight
        return float(fitness_value)

    def evaluate_population(self, pop: np.ndarray):
        pop_guarded = np.array([self.apply_guardrails(p) for p in pop])
        X = self._features_for_prices(pop_guarded)
        try:
            preds = np.array(self.predictor.predict(X))
        except Exception:
            preds = np.array(self.predictor.predict(X.values))
        preds = np.maximum(0.0, preds)
        profits = (pop_guarded - self.today["cost"]) * preds
        return profits, pop_guarded, preds

    def initialize_population(self) -> np.ndarray:
        raw = self.rng.uniform(self.low, self.high, size=self.pop_size)
        return np.round(raw / self.price_step) * self.price_step

    def tournament_selection(self, pop: np.ndarray, fitnesses: np.ndarray, k: int = 3) -> np.ndarray:
        idx = []
        for _ in range(len(pop)):
            competitors = self.rng.integers(0, len(pop), size=k)
            winner = competitors[np.argmax(fitnesses[competitors])]
            idx.append(winner)
        return pop[idx]

    def crossover(self, parents: np.ndarray) -> np.ndarray:
        children = parents.copy()
        for i in range(0, len(parents)-1, 2):
            if self.rng.random() < self.cx_prob:
                p1 = parents[i]; p2 = parents[i+1]
                alpha = self.rng.random()
                c1 = alpha*p1 + (1-alpha)*p2
                c2 = alpha*p2 + (1-alpha)*p1
                children[i] = np.round(c1 / self.price_step) * self.price_step
                children[i+1] = np.round(c2 / self.price_step) * self.price_step
        return children

    def mutate(self, pop: np.ndarray) -> np.ndarray:
        scale = 0.10 * (self.high - self.low)
        for i in range(len(pop)):
            if self.rng.random() < self.mut_prob:
                pop[i] += self.rng.normal(0, scale)
                pop[i] = min(max(pop[i], self.low), self.high)
                pop[i] = np.round(pop[i] / self.price_step) * self.price_step
        return pop

    def run(self) -> Dict[str, Any]:
        pop = self.initialize_population()
        best_history = []
        mean_history = []
        all_best_individuals = []

        profits, pop_guarded, preds = self.evaluate_population(pop)
        for gen in range(self.generations):
            gen_weight = (self.r2_alpha + self.r2_beta * max(0.0, self.avg_cv_r2))
            fitnesses = profits * gen_weight
            elite_idx = np.argsort(-fitnesses)[:self.elite_size]
            elites = pop[elite_idx].copy()
            selected = self.tournament_selection(pop, fitnesses, k=3)
            children = self.crossover(selected)
            children = self.mutate(children)
            pop = children
            pop[:self.elite_size] = elites
            profits, pop_guarded, preds = self.evaluate_population(pop)
            fitnesses = profits * gen_weight
            best_idx = int(np.argmax(fitnesses))
            best_price = float(pop_guarded[best_idx])
            best_profit = float(profits[best_idx])
            mean_profit = float(np.mean(profits))
            best_history.append(best_profit); mean_history.append(mean_profit)
            all_best_individuals.append({"gen": gen, "best_price": best_price, "best_profit": best_profit})
            if (gen % 5 == 0) or (gen == self.generations-1):
                print(f"GA gen {gen+1}/{self.generations}: best_price={best_price:.2f}, best_profit={best_profit:.2f}, mean_profit={mean_profit:.2f}, gen_weight={gen_weight:.3f}")

        profits, pop_guarded, preds = self.evaluate_population(pop)
        gen_weight = (self.r2_alpha + self.r2_beta * max(0.0, self.avg_cv_r2))
        fitnesses = profits * gen_weight
        best_idx = int(np.argmax(fitnesses))
        best_price_raw = float(pop[best_idx])
        best_price_guarded = float(pop_guarded[best_idx])
        best_volume = float(preds[best_idx])
        best_profit = float(profits[best_idx])
        history_df = pd.DataFrame(all_best_individuals)
        pop_final_df = pd.DataFrame({"price_raw": pop, "price_guarded": pop_guarded, "pred_volume": preds, "profit": profits})
        pop_final_df = pop_final_df.sort_values("profit", ascending=False).reset_index(drop=True)
        return {
            "best_price_raw": best_price_raw,
            "best_price_guarded": best_price_guarded,
            "best_volume": best_volume,
            "best_profit": best_profit,
            "history": history_df,
            "pop_final": pop_final_df,
            "best_history": best_history,
            "mean_history": mean_history
        }

# ---------------------------
# Metrics & plotting helpers
# ---------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mask = (y_true != 0)
    mape = float(np.nan) if mask.sum() == 0 else (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100.0
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), "mape_pct": float(mape)}

def plot_candidates_grid(df_candidates: pd.DataFrame, out_file: str, recommended: float = None):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df_candidates["candidate_price"], df_candidates["expected_profit"], marker='o')
    if recommended is not None:
        ax.axvline(recommended, color='red', linestyle='--', label=f"recommended {recommended:.2f}")
        ax.legend()
    ax.set_xlabel("Candidate price")
    ax.set_ylabel("Expected profit")
    ax.set_title("Profit vs price grid")
    save_plot(fig, out_file)

def plot_ga_history(history_df: pd.DataFrame, out_file: str):
    fig, ax = plt.subplots(figsize=(8,4))
    if not history_df.empty:
        ax.plot(history_df["gen"] + 1, history_df["best_profit"], marker='o')
    ax.set_xlabel("Generation"); ax.set_ylabel("Best profit"); ax.set_title("GA best profit per generation")
    save_plot(fig, out_file)

def plot_feature_importances(series: pd.Series, out_file: str, top_n: int = 25):
    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * min(len(series), top_n))))
    series.head(top_n).sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Feature importances"); ax.set_xlabel("Importance")
    save_plot(fig, out_file)

def plot_pred_vs_actual(y_true_train, y_pred_train, y_true_test, y_pred_test, out_file):
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    axes[0].scatter(y_true_train, y_pred_train, alpha=0.4); axes[0].set_title("Train: predicted vs actual"); axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
    axes[1].scatter(y_true_test, y_pred_test, alpha=0.4); axes[1].set_title("Test: predicted vs actual"); axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Predicted")
    save_plot(fig, out_file)

# ---------------------------
# Build today's base features for candidate eval
# ---------------------------

def build_today_base_features(df_history: pd.DataFrame, today: Dict[str, Any]) -> Dict[str, Any]:
    last = df_history.sort_values("date").iloc[-1]
    base = {
        "cost": today["cost"],
        "comp_mean": today["comp_mean"],
        "comp_min": today["comp_min"],
        "comp_max": today["comp_max"],
        "comp_spread": today["comp_max"] - today["comp_min"],
        "dow": int(pd.Timestamp(today["date"]).dayofweek),
        "month": int(pd.Timestamp(today["date"]).month),
        "week": int(pd.Timestamp(today["date"]).isocalendar().week),
        "dayofyear": int(pd.Timestamp(today["date"]).dayofyear),
        "is_weekend": int(pd.Timestamp(today["date"]).dayofweek >= 5),
        "trend": len(df_history) + 1,
        "volume_lag1": float(last["volume"]),
        "volume_lag7": float(df_history.iloc[-7]["volume"]) if len(df_history) >= 7 else float(last["volume"]),
        "volume_roll7": float(df_history["volume"].tail(7).mean()) if len(df_history) >= 7 else float(df_history["volume"].mean()),
        "price_lag1": float(last["price"]),
        "price_lag7": float(df_history.iloc[-7]["price"]) if len(df_history) >= 7 else float(last["price"]),
        "price_roll7": float(df_history["price"].tail(7).mean()) if len(df_history) >= 7 else float(df_history["price"].mean()),
        "price_gap_mean_lag1": float((last["price"] - last[["comp1_price","comp2_price","comp3_price"]].mean())),
        "price_gap_mean_lag7": float((df_history.iloc[-7]["price"] - df_history.iloc[-7][["comp1_price","comp2_price","comp3_price"]].mean())) if len(df_history) >= 7 else 0.0,
        "price_gap_mean_roll7": float(((df_history["price"] - df_history[["comp1_price","comp2_price","comp3_price"]].mean(axis=1))).tail(7).mean()) if len(df_history) >= 7 else 0.0
    }
    return base

# ---------------------------
# Main script
# ---------------------------

def main():
    p = argparse.ArgumentParser(description="Price optimizer using stacked ensemble (LightGBM/XGBoost/RF) + GA")
    p.add_argument("--data", default="data/oil_retail_history.xlsx")
    p.add_argument("--today", default="data/today_example.json")
    p.add_argument("--out", "--outdir", dest="outdir", default="outputs")
    p.add_argument("--top_k", type=int, default=15, help="Top-K features to keep")
    p.add_argument("--test_size", type=float, default=0.2, help="Chronological test fraction")
    p.add_argument("--cv_splits", type=int, default=3, help="TimeSeriesSplit folds for CV")
    p.add_argument("--opt_trials", type=int, default=30, help="Optuna trials per model")
    args = p.parse_args()

    outdir = args.outdir
    ensure_dir(outdir)

    print("Loading data...")
    df = load_history(args.data)
    today = load_today(args.today)
    data_summary(df)

    print("Feature engineering...")
    df_feat = build_features(df, with_lags=True)
    print(f"Rows after features: {len(df_feat)}")

    train_df, test_df = time_train_test_split(df_feat, test_size=args.test_size)
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    candidate_features = [
        "price","cost","comp_mean","comp_min","comp_max","price_gap_mean",
        "comp_spread","margin",
        "dow","month","week","dayofyear","is_weekend","trend",
        "volume_lag1","volume_lag7","volume_roll7",
        "price_lag1","price_lag7","price_roll7",
        "price_gap_mean_lag1","price_gap_mean_lag7","price_gap_mean_roll7"
    ]
    candidate_features = [f for f in candidate_features if f in train_df.columns]

    X_train_df = train_df[candidate_features].copy()
    y_train = train_df["volume"].astype(float).copy()
    X_test_df = test_df[candidate_features].copy()
    y_test = test_df["volume"].astype(float).copy()

    print("Selecting top features...")
    top_k = min(args.top_k, len(candidate_features))
    selected = select_top_features_by_importance(X_train_df, y_train, top_k=top_k)
    print("Selected features:", selected)

    # Prepare final matrices for model training
    X_train_sel = X_train_df[selected].fillna(0.0)
    X_test_sel = X_test_df[selected].fillna(0.0)

    # Scale features (fit on train)
    scaler = StandardScaler()
    scaler.fit(X_train_sel.values)
    X_train_scaled = scaler.transform(X_train_sel.values)
    X_test_scaled = scaler.transform(X_test_sel.values)

    # ---------- Optuna tuning ----------
    print("Starting Optuna tuning...")

    best_lgb_params = None
    best_xgb_params = None

    if HAS_LGB:
        print("Tuning LightGBM...")
        study_lgb = optuna.create_study(direction="minimize")
        func = lambda trial: objective_lgb(trial, X_train_sel, y_train, tss_splits=args.cv_splits)
        study_lgb.optimize(func, n_trials=args.opt_trials, show_progress_bar=False)
        best_lgb_params = study_lgb.best_params
        print("Best LightGBM params:", best_lgb_params)
    else:
        print("LightGBM not available — skipping LGB tuning.")

    if HAS_XGB:
        print("Tuning XGBoost...")
        study_xgb = optuna.create_study(direction="minimize")
        func2 = lambda trial: objective_xgb(trial, X_train_sel, y_train, tss_splits=args.cv_splits)
        study_xgb.optimize(func2, n_trials=args.opt_trials, show_progress_bar=False)
        best_xgb_params = study_xgb.best_params
        best_xgb_params["objective"] = "reg:squarederror"
        best_xgb_params["random_state"] = 42
        print("Best XGBoost params:", best_xgb_params)
    else:
        print("XGBoost not available — skipping XGB tuning.")

    # ---------- Build base learners ----------
    estimators = []
    if HAS_LGB and best_lgb_params is not None:
        lgb_params = best_lgb_params.copy()
        lgb_params["random_state"] = 42
        lgb = LGBMRegressor(**lgb_params)
        estimators.append(("lgb", lgb))

    if HAS_XGB and best_xgb_params is not None:
        xgb_model = XGBRegressor(**best_xgb_params)
        estimators.append(("xgb", xgb_model))

    # Always include RandomForest as a robust tree-based model
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    estimators.append(("rf", rf))

    if len(estimators) == 0:
        raise RuntimeError("No estimators available (install xgboost or lightgbm).")

    # Stacking regressor with Ridge meta-learner.
    final_estimator = Ridge(alpha=1.0)
    # Wrap stacking in pipeline with scaler so predict receives scaled input
    stacking = StackingRegressor(estimators=estimators, final_estimator=final_estimator, n_jobs=-1, passthrough=False)

    # Fit stacking on scaled data
    print("Training stacked ensemble...")
    stacking.fit(X_train_scaled, y_train)

    # Wrap predictor
    predictor = EnsemblePredictor(model=stacking, scaler=scaler, selected_features=selected, df_history=df_feat)

    # Evaluate performance on train/test
    y_pred_train = stacking.predict(X_train_scaled)
    y_pred_test = stacking.predict(X_test_scaled)
    metrics_train = compute_metrics(y_train.values, y_pred_train)
    metrics_test = compute_metrics(y_test.values, y_pred_test)
    print("TRAIN metrics:", metrics_train)
    print("TEST metrics:", metrics_test)

    # Save plots & feature importances
    try:
        # If LGB or XGB present, get feature importances by averaging available importances
        importances = pd.Series(0.0, index=selected)
        cnt = 0
        if HAS_LGB and 'lgb' in dict(estimators):
            imp = pd.Series(stacking.named_estimators_['lgb'].feature_importances_, index=selected)
            importances += imp; cnt += 1
        if HAS_XGB and 'xgb' in dict(estimators):
            imp = pd.Series(stacking.named_estimators_['xgb'].feature_importances_, index=selected)
            importances += imp; cnt += 1
        # rf importance
        if 'rf' in dict(estimators):
            imp = pd.Series(stacking.named_estimators_['rf'].feature_importances_, index=selected)
            importances += imp; cnt += 1
        if cnt > 0:
            importances /= cnt
            plot_feature_importances(importances.sort_values(ascending=False), os.path.join(outdir, "feature_importances.png"))
            Path(os.path.join(outdir, "feature_importances.csv")).write_text(importances.sort_values(ascending=False).to_csv())
    except Exception as e:
        print("Failed to produce feature importances:", e)

    plot_pred_vs_actual(y_train.values, y_pred_train, y_test.values, y_pred_test, os.path.join(outdir, "pred_vs_actual.png"))

    # ---------- Compute cross-validated R2 using stacking with TimeSeriesSplit ----------
    def compute_cv_r2_stack(X_full: pd.DataFrame, y_full: pd.Series, model_pipeline, n_splits=3):
        tss = TimeSeriesSplit(n_splits=n_splits)
        r2s = []
        for tr_idx, val_idx in tss.split(X_full):
            Xtr = scaler.transform(X_full.iloc[tr_idx][selected].values)
            Xval = scaler.transform(X_full.iloc[val_idx][selected].values)
            ytr, yval = y_full.iloc[tr_idx], y_full.iloc[val_idx]
            model_pipeline.fit(Xtr, ytr)
            preds = model_pipeline.predict(Xval)
            r2s.append(r2_score(yval, preds))
        if len(r2s) == 0:
            return 0.0
        return float(np.mean(r2s))

    avg_cv_r2 = compute_cv_r2_stack(pd.concat([X_train_sel, X_test_sel], ignore_index=True), pd.concat([y_train, y_test], ignore_index=True), stacking, n_splits=args.cv_splits)
    print(f"Average CV R2 (stacking): {avg_cv_r2:.4f}")

    # ---------- Baseline grid sweep ----------
    grid = candidate_price_grid(today, step=0.05)
    base = build_today_base_features(df, today)
    rows = []
    for p_val in grid:
        feat = base.copy(); feat["price"] = float(p_val); feat["margin"] = float(p_val - today["cost"]); feat["price_gap_mean"] = float(p_val - base["comp_mean"])
        X_row = pd.DataFrame([feat])
        for col in selected:
            if col not in X_row.columns:
                X_row[col] = 0.0
        X_row_sel = X_row[selected].fillna(0.0)
        X_row_scaled = scaler.transform(X_row_sel.values)
        try:
            v = float(max(0.0, stacking.predict(X_row_scaled)[0]))
        except Exception:
            v = float(0.0)
        profit = (p_val - today["cost"]) * v
        rows.append({"candidate_price": p_val, "pred_volume": v, "expected_profit": profit})
    df_grid = pd.DataFrame(rows).sort_values("expected_profit", ascending=False).reset_index(drop=True)
    save_excel(df_grid, os.path.join(outdir, "candidate_grid_reference.xlsx"))
    plot_candidates_grid(df_grid, os.path.join(outdir, "profit_grid_reference.png"))

    # ---------- Run GA (uses predictor wrapper) ----------
    ga = GeneticOptimizer(
        predictor=predictor,
        df_history=df,
        today=today,
        selected_features=selected,
        avg_cv_r2=avg_cv_r2,
        r2_alpha=0.7,
        r2_beta=0.3,
        pop_size=100,
        generations=80,
        crossover_prob=0.6,
        mutation_prob=0.25,
        elite_size=6,
        price_step=0.01,
        seed=42
    )

    ga_res = ga.run()
    print("\nGA result:")
    print(f"Best raw price: {ga_res['best_price_raw']:.2f}")
    print(f"Best guarded price: {ga_res['best_price_guarded']:.2f}")
    print(f"Predicted volume: {ga_res['best_volume']:.1f}")
    print(f"Predicted profit: {ga_res['best_profit']:.2f}")

    save_excel(ga_res["pop_final"], os.path.join(outdir, "ga_population_final.xlsx"))
    ga_res["history"].to_csv(os.path.join(outdir, "ga_history.csv"), index=False)
    plot_ga_history(ga_res["history"], os.path.join(outdir, "ga_history.png"))

    recommended_price = ga_res["best_price_guarded"]
    recommendation = {
        "recommended_price": recommended_price,
        "best_price_raw": ga_res["best_price_raw"],
        "pred_volume": ga_res["best_volume"],
        "expected_profit": ga_res["best_profit"],
        "today": {
            "date": str(today["date"].date()),
            "yesterday_price": today["price"],
            "cost": today["cost"],
            "comp_mean": today["comp_mean"]
        },
        "model_metrics": {"train": metrics_train, "test": metrics_test, "avg_cv_r2": avg_cv_r2},
        "selected_features": selected
    }
    save_json(recommendation, os.path.join(outdir, "recommendation_stack_ga.json"))

    # Save models & scaler
    try:
        joblib.dump(stacking, os.path.join(outdir, "stacking_model.pkl"))
        joblib.dump(scaler, os.path.join(outdir, "scaler.pkl"))
        print("Saved stacking model and scaler.")
    except Exception as e:
        print("Failed to save model/scaler:", e)

    print("\nAll outputs saved in:", outdir)
    print("Done.")

if __name__ == "__main__":
    main()
