#!/usr/bin/env python3
"""
ml_price_optimizer_ga.py

Full pipeline for Fuel Price Optimization using a Genetic Algorithm to maximize
profit = (price - cost) * predicted_volume, with an R²-based generalization
regularizer baked into the GA fitness.

- Loads historical data (Excel/CSV) and today's JSON
- Builds features (calendar, competitor aggregates, margin, lags)
- Performs feature selection
- Trains a demand model (GradientBoosting or XGBoost)
- Computes cross-validated R² (TimeSeriesSplit) as a measure of generalization
- Runs a Genetic Algorithm to search the continuous price space for the price
  that yields maximum expected profit *and* better generalization
- Saves Excel, PNG plots, recommendation JSON, and CSVs

Usage example:
python ml_price_optimizer_ga.py --data data/oil_retail_history.xlsx --today data/today_example.json --out outputs --model gbr --pop 120 --gens 80

Author: ChatGPT
Date: 2025
"""
import argparse
import json
import os
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# optional xgboost
try:
    import xgboost as xgb  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

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

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date")
    df["volume_lag1"] = df["volume"].shift(1)
    df["volume_lag7"] = df["volume"].shift(7)
    df["volume_roll7"] = df["volume"].shift(1).rolling(7).mean()
    df["price_lag1"] = df["price"].shift(1)
    df["price_lag7"] = df["price"].shift(7)
    df["price_roll7"] = df["price"].shift(1).rolling(7).mean()
    df["price_gap_mean_lag1"] = df["price_gap_mean"].shift(1)
    df["price_gap_mean_lag7"] = df["price_gap_mean"].shift(7)
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
# Train/test split & selection
# ---------------------------

def time_train_test_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n < 10:
        raise ValueError("Not enough data for reliable split")
    split_at = int(n * (1.0 - test_size))
    train = df.iloc[:split_at].reset_index(drop=True)
    test = df.iloc[split_at:].reset_index(drop=True)
    return train, test

def select_top_features_by_importance(X_train: pd.DataFrame, y_train: pd.Series, top_k: int = 12) -> List[str]:
    model = GradientBoostingRegressor(random_state=42, n_estimators=150)
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
    print("\nFeature importances (top):")
    print(feat_imp.head(top_k).to_string())
    return list(feat_imp.head(top_k).index)

# ---------------------------
# Demand model training
# ---------------------------

def train_gbr(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[GradientBoostingRegressor, Dict[str, Any]]:
    model = GradientBoostingRegressor(random_state=42, n_estimators=250)
    model.fit(X_train, y_train)
    return model, {"type":"gbr"}

def train_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, Dict[str, Any]]:
    if not HAS_XGB:
        raise ImportError("xgboost not installed. Install or use --model xgb")
    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_estimators=250)
    model.fit(X_train, y_train)
    return model, {"type":"xgb"}

def compute_cv_r2(X: pd.DataFrame, y: pd.Series, n_splits: int = 3) -> float:
    """
    Compute average TimeSeriesSplit R² for given X,y. Used as a stable
    measure of model generalization (surrogate model).
    """
    tss = TimeSeriesSplit(n_splits=n_splits)
    r2s = []
    for train_idx, val_idx in tss.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        m = GradientBoostingRegressor(random_state=42, n_estimators=150)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_val)
        r2s.append(r2_score(y_val, preds))
    return float(np.mean(r2s))

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
# Genetic Algorithm implementation
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
        # ensure minimum spread
        if high - low < 0.5:
            low = comp_mean - 0.5
            high = comp_mean + 0.5
        # add some margin for exploration
        self.low = round(low - 0.5, 2)
        self.high = round(high + 0.5, 2)

        # guardrail params
        self.max_daily_change = 1.0
        self.min_margin = 0.10
        self.keep_within_competitors = True
        self.comp_band = 0.75

    def apply_guardrails(self, p: float) -> float:
        # ensures guardrails are applied consistently within GA
        p = float(p)
        p = max(p, self.today["cost"] + self.min_margin)
        p = min(p, self.today["price"] + self.max_daily_change)
        p = max(p, self.today["price"] - self.max_daily_change)
        if self.keep_within_competitors:
            p = min(p, self.today["comp_mean"] + self.comp_band)
            p = max(p, self.today["comp_mean"] - self.comp_band)
        return round(p, 2)

    def _features_for_prices(self, prices: np.ndarray) -> pd.DataFrame:
        """Construct a DataFrame of features (selected_features order) for batch of prices."""
        base = build_today_base_features(self.df_history, self.today)
        rows = []
        for p in prices:
            feat = base.copy()
            feat["price"] = float(p)
            feat["margin"] = float(p - base["cost"])
            feat["price_gap_mean"] = float(p - base["comp_mean"])
            rows.append(feat)
        X = pd.DataFrame(rows)
        # ensure all selected_features present
        for c in self.selected_features:
            if c not in X.columns:
                X[c] = 0.0
        X = X[self.selected_features]
        return X

    def fitness(self, price: float) -> float:
        """
        Fitness = profit * (r2_alpha + r2_beta * avg_cv_r2)
        avg_cv_r2 is computed once outside (cheap) and passed in.
        """
        p_guarded = self.apply_guardrails(price)
        X = self._features_for_prices(np.array([p_guarded]))
        # predictor may accept dataframe or array
        try:
            v = float(self.predictor.predict(X)[0])
        except Exception:
            v = float(self.predictor.predict(X.values)[0])
        v = max(0.0, v)
        profit = (p_guarded - self.today["cost"]) * v

        # weight by generalization factor (clip avg_cv_r2 to >=0)
        gen_weight = (self.r2_alpha + self.r2_beta * max(0.0, self.avg_cv_r2))
        fitness_value = profit * gen_weight
        return float(fitness_value)

    def evaluate_population(self, pop: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Evaluate array of prices efficiently in batch
        pop_guarded = np.array([self.apply_guardrails(p) for p in pop])
        X = self._features_for_prices(pop_guarded)
        # Predict in batch
        try:
            preds = np.array(self.predictor.predict(X))
        except Exception:
            preds = np.array(self.predictor.predict(X.values))
        preds = np.maximum(0.0, preds)
        profits = (pop_guarded - self.today["cost"]) * preds
        # fitness uses generalization multiplier but we keep profits separate
        return profits, pop_guarded, preds

    def initialize_population(self) -> np.ndarray:
        # uniform sampling across [low, high]
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
        # arithmetic crossover: children are average +/- small random factor
        children = parents.copy()
        for i in range(0, len(parents)-1, 2):
            if self.rng.random() < self.cx_prob:
                p1 = parents[i]
                p2 = parents[i+1]
                alpha = self.rng.random()
                c1 = alpha*p1 + (1-alpha)*p2
                c2 = alpha*p2 + (1-alpha)*p1
                children[i] = np.round(c1 / self.price_step) * self.price_step
                children[i+1] = np.round(c2 / self.price_step) * self.price_step
        return children

    def mutate(self, pop: np.ndarray) -> np.ndarray:
        # Add gaussian noise; scale = 10% of price range
        scale = 0.10 * (self.high - self.low)
        for i in range(len(pop)):
            if self.rng.random() < self.mut_prob:
                pop[i] += self.rng.normal(0, scale)
                # clamp
                pop[i] = min(max(pop[i], self.low), self.high)
                pop[i] = np.round(pop[i] / self.price_step) * self.price_step
        return pop

    def run(self) -> Dict[str, Any]:
        pop = self.initialize_population()
        best_history = []
        mean_history = []
        all_best_individuals = []

        # evaluate initial
        profits, pop_guarded, preds = self.evaluate_population(pop)
        # convert profits to fitness by multiplying with gen_weight
        for gen in range(self.generations):
            # fitness values (using generalization multiplier)
            gen_weight = (self.r2_alpha + self.r2_beta * max(0.0, self.avg_cv_r2))
            fitnesses = profits * gen_weight

            # Elitism: keep top elites
            elite_idx = np.argsort(-fitnesses)[:self.elite_size]
            elites = pop[elite_idx].copy()

            # Selection (tournament)
            selected = self.tournament_selection(pop, fitnesses, k=3)

            # Crossover
            children = self.crossover(selected)

            # Mutation
            children = self.mutate(children)

            # Next gen: replace first elite_size by elites
            pop = children
            pop[:self.elite_size] = elites

            # Evaluate
            profits, pop_guarded, preds = self.evaluate_population(pop)
            # fitness with generalization multiplier
            fitnesses = profits * gen_weight
            best_idx = int(np.argmax(fitnesses))
            best_price = float(pop_guarded[best_idx])
            best_profit = float(profits[best_idx])
            mean_profit = float(np.mean(profits))
            best_history.append(best_profit)
            mean_history.append(mean_profit)
            all_best_individuals.append({"gen": gen, "best_price": best_price, "best_profit": best_profit})

            # logging
            if (gen % 5 == 0) or (gen == self.generations-1):
                print(f"GA gen {gen+1}/{self.generations}: best_price={best_price:.2f}, best_profit={best_profit:.2f}, mean_profit={mean_profit:.2f}, gen_weight={gen_weight:.3f}")

        # final evaluation
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
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best profit")
    ax.set_title("GA best profit per generation")
    save_plot(fig, out_file)

def plot_feature_importances(series: pd.Series, out_file: str, top_n: int = 25):
    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * min(len(series), top_n))))
    series.head(top_n).sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Feature importances")
    ax.set_xlabel("Importance")
    save_plot(fig, out_file)

def plot_pred_vs_actual(y_true_train, y_pred_train, y_true_test, y_pred_test, out_file):
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    axes[0].scatter(y_true_train, y_pred_train, alpha=0.4)
    axes[0].set_title("Train: predicted vs actual"); axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
    axes[1].scatter(y_true_test, y_pred_test, alpha=0.4)
    axes[1].set_title("Test: predicted vs actual"); axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Predicted")
    save_plot(fig, out_file)

# ---------------------------
# Candidate grid function (for reference)
# ---------------------------

def candidate_price_grid(today: Dict[str, Any], low: float = None, high: float = None, step: float = 0.05) -> np.ndarray:
    comp_mean = today["comp_mean"]
    base_low = max(today["cost"] + 0.01, comp_mean - 2.0)
    base_high = comp_mean + 2.0
    if low is not None:
        base_low = max(base_low, low)
    if high is not None:
        base_high = min(base_high, high)
    if base_high - base_low < 0.5:
        base_low = min(base_low, comp_mean - 0.5)
        base_high = max(base_high, comp_mean + 0.5)
    grid = np.round(np.arange(base_low, base_high + 1e-9, step), 2)
    return grid

# ---------------------------
# Main script
# ---------------------------

def main():
    p = argparse.ArgumentParser(description="Fuel Price Optimizer using Genetic Algorithm (with R² regularization)")
    p.add_argument("--data", default="data/oil_retail_history.xlsx")
    p.add_argument("--today", default="data/today_example.json")
    p.add_argument("--out", "--outdir", dest="outdir", default="outputs")
    p.add_argument("--model", choices=["gbr","xgb"], default="gbr", help="Surrogate demand model")
    p.add_argument("--pop", type=int, default=120, help="GA population size")
    p.add_argument("--gens", type=int, default=80, help="GA generations")
    p.add_argument("--cx", type=float, default=0.6, help="GA crossover probability")
    p.add_argument("--mut", type=float, default=0.25, help="GA mutation probability")
    p.add_argument("--elite", type=int, default=6, help="GA elite size")
    p.add_argument("--step", type=float, default=0.05, help="price grid step for reference sweep")
    p.add_argument("--top_k", type=int, default=12, help="Top-K features to keep")
    p.add_argument("--test_size", type=float, default=0.2, help="Chronological test fraction")
    p.add_argument("--r2_alpha", type=float, default=0.7, help="Fitness baseline weight")
    p.add_argument("--r2_beta", type=float, default=0.3, help="Fitness multiplier for avg_cv_r2")
    p.add_argument("--cv_splits", type=int, default=3, help="TimeSeriesSplit folds for CV R²")
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

    X_train = train_df[candidate_features].copy()
    y_train = train_df["volume"].astype(float).copy()
    X_test = test_df[candidate_features].copy()
    y_test = test_df["volume"].astype(float).copy()

    # Feature selection
    print("Selecting top features...")
    top_k = min(args.top_k, len(candidate_features))
    selected = select_top_features_by_importance(X_train, y_train, top_k=top_k)
    print("Selected features:", selected)

    X_train_sel = X_train[selected].copy()
    X_test_sel = X_test[selected].copy()

    # Train surrogate demand model
    if args.model == "gbr":
        print("Training GradientBoosting surrogate demand model...")
        predictor, meta = train_gbr(X_train_sel, y_train)
    else:
        print("Training XGBoost surrogate demand model...")
        if not HAS_XGB:
            raise RuntimeError("xgboost not installed. Install xgboost or use --model gbr")
        predictor, meta = train_xgb(X_train_sel, y_train)

    # Evaluate demand model performance
    print("Evaluating surrogate model...")
    y_pred_train = predictor.predict(X_train_sel)
    y_pred_test = predictor.predict(X_test_sel)
    metrics_train = compute_metrics(y_train.values, y_pred_train)
    metrics_test = compute_metrics(y_test.values, y_pred_test)
    print("TRAIN metrics:", metrics_train)
    print("TEST metrics:", metrics_test)

    # Compute cross-validated R² (TimeSeriesSplit) on training data
    print(f"Computing cross-validated R² with {args.cv_splits} splits...")
    avg_cv_r2 = compute_cv_r2(X_train_sel, y_train, n_splits=args.cv_splits)
    print(f"Average CV R² (train folds): {avg_cv_r2:.4f}")

    # Save predicted vs actual plot
    plot_pred_vs_actual(y_train.values, y_pred_train, y_test.values, y_pred_test, os.path.join(outdir, "pred_vs_actual.png"))

    # Feature importances
    try:
        importances = None
        if hasattr(predictor, "feature_importances_"):
            importances = pd.Series(predictor.feature_importances_, index=selected).sort_values(ascending=False)
        elif HAS_XGB and isinstance(predictor, xgb.XGBRegressor):
            importances = pd.Series(predictor.feature_importances_, index=selected).sort_values(ascending=False)
        if importances is not None:
            plot_feature_importances(importances, os.path.join(outdir, "feature_importances.png"))
            Path(os.path.join(outdir, "feature_importances.csv")).write_text(importances.to_csv())
            print("Saved feature importances.")
    except Exception as e:
        print("Failed to save feature importances:", e)

    # Baseline grid sweep (for reference)
    grid = candidate_price_grid(today, step=args.step)
    base = build_today_base_features(df, today)
    rows = []
    for p_val in grid:
        feat = base.copy()
        feat["price"] = float(p_val)
        feat["margin"] = float(p_val - today["cost"])
        feat["price_gap_mean"] = float(p_val - base["comp_mean"])
        X_row = pd.DataFrame([feat])
        for col in selected:
            if col not in X_row.columns:
                X_row[col] = 0.0
        X_row = X_row[selected]
        v = float(max(0.0, predictor.predict(X_row)[0]))
        profit = (p_val - today["cost"]) * v
        rows.append({"candidate_price": p_val, "pred_volume": v, "expected_profit": profit})
    df_grid = pd.DataFrame(rows).sort_values("expected_profit", ascending=False).reset_index(drop=True)
    save_excel(df_grid, os.path.join(outdir, "candidate_grid_reference.xlsx"))
    plot_candidates_grid(df_grid, os.path.join(outdir, "profit_grid_reference.png"))

    # Run Genetic Algorithm to search continuous price
    ga = GeneticOptimizer(
        predictor=predictor,
        df_history=df,
        today=today,
        selected_features=selected,
        avg_cv_r2=avg_cv_r2,
        r2_alpha=args.r2_alpha,
        r2_beta=args.r2_beta,
        pop_size=args.pop,
        generations=args.gens,
        crossover_prob=args.cx,
        mutation_prob=args.mut,
        elite_size=args.elite,
        price_step=0.01,
        seed=42
    )

    ga_res = ga.run()
    print("\nGA result:")
    print(f"Best raw price: {ga_res['best_price_raw']:.2f}")
    print(f"Best guarded price: {ga_res['best_price_guarded']:.2f}")
    print(f"Predicted volume: {ga_res['best_volume']:.1f}")
    print(f"Predicted profit: {ga_res['best_profit']:.2f}")

    # Save GA outputs
    save_excel(ga_res["pop_final"], os.path.join(outdir, "ga_population_final.xlsx"))
    ga_res["history"].to_csv(os.path.join(outdir, "ga_history.csv"), index=False)
    plot_ga_history(ga_res["history"], os.path.join(outdir, "ga_history.png"))

    # Final recommended after ensuring guardrails
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
    save_json(recommendation, os.path.join(outdir, "recommendation_ga.json"))

    print("\nAll outputs saved in:", outdir)
    print("Done.")

if __name__ == "__main__":
    main()
