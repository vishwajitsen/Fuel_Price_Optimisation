#!/usr/bin/env python3
"""
ml_price_optimizer.py

Self-contained ML (Gradient Boosting) price optimizer.
Saves outputs into the specified output directory.

Usage:
python ml_price_optimizer.py --data data/oil_retail_history.xlsx --today data/today_example.json --out outputs --tune False --top_k 12
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
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------
# Utilities
# ---------------------------

def ensure_dir(p: str):
    if p:
        Path(p).mkdir(parents=True, exist_ok=True)

def save_json(obj: Dict[str, Any], out_file: str):
    ensure_dir(os.path.dirname(out_file))
    Path(out_file).write_text(json.dumps(obj, indent=2))
    print(f"Saved JSON output: {out_file}")

def save_candidates_to_excel(df_candidates: pd.DataFrame, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    df_candidates.to_excel(out_path, index=False)
    print(f"Saved candidate table to Excel: {out_path}")

def save_plot(fig, out_file: str):
    ensure_dir(os.path.dirname(out_file))
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_file}")

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
    today["comp_mean"] = np.mean([today["comp1_price"], today["comp2_price"], today["comp3_price"]])
    today["comp_min"] = min(today["comp1_price"], today["comp2_price"], today["comp3_price"])
    today["comp_max"] = max(today["comp1_price"], today["comp2_price"], today["comp3_price"])
    return today

# ---------------------------
# Exploratory summary (console)
# ---------------------------

def data_summary(df: pd.DataFrame):
    print("\n--- DATA SUMMARY ---")
    print("Rows, Columns:", df.shape)
    print("Date range:", df["date"].min().date(), "to", df["date"].max().date())
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
# Candidate grid & guardrails
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

def apply_guardrails(rec_price: float, today: Dict[str, Any], max_daily_change: float = 1.0, min_margin: float = 0.10, keep_within_competitors: bool = True, comp_band: float = 0.75) -> float:
    p = float(rec_price)
    p = max(p, today["cost"] + min_margin)
    p = min(p, today["price"] + max_daily_change)
    p = max(p, today["price"] - max_daily_change)
    if keep_within_competitors:
        p = min(p, today["comp_mean"] + comp_band)
        p = max(p, today["comp_mean"] - comp_band)
    return round(p, 2)

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
    print("\nTop feature importances:")
    print(feat_imp.head(top_k).to_string())
    return list(feat_imp.head(top_k).index)

# ---------------------------
# Model training & evaluation
# ---------------------------

def train_gbr(X_train: pd.DataFrame, y_train: pd.Series, tune: bool = False) -> Tuple[GradientBoostingRegressor, Dict[str, Any]]:
    meta = {}
    if tune:
        print("Running GridSearchCV (time-series folds)...")
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1]
        }
        tscv = TimeSeriesSplit(n_splits=3)
        base = GradientBoostingRegressor(random_state=42)
        gs = GridSearchCV(base, param_grid, scoring="neg_mean_absolute_error", cv=tscv, n_jobs=-1)
        gs.fit(X_train, y_train)
        meta["best_params"] = gs.best_params_
        model = gs.best_estimator_
        print("Best params:", gs.best_params_)
    else:
        model = GradientBoostingRegressor(random_state=42, n_estimators=200)
        model.fit(X_train, y_train)
    return model, meta

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = float(np.nan) if mask.sum() == 0 else (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100.0
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), "mape_pct": float(mape)}

# ---------------------------
# Build today's base features and evaluate grid
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

def evaluate_candidate_grid(model, selected_features: List[str], df_history: pd.DataFrame, today: Dict[str, Any], step: float = 0.05) -> Tuple[pd.DataFrame, float]:
    grid = candidate_price_grid(today, step=step)
    base = build_today_base_features(df_history, today)
    rows = []
    for p in grid:
        feat = base.copy()
        feat["price"] = float(p)
        feat["margin"] = float(p - today["cost"])
        feat["price_gap_mean"] = float(p - base["comp_mean"])
        X_row = pd.DataFrame([feat])
        for col in selected_features:
            if col not in X_row.columns:
                X_row[col] = 0.0
        X_row = X_row[selected_features]
        v = float(model.predict(X_row)[0])
        v = max(0.0, v)
        profit = (p - today["cost"]) * v
        rows.append({"candidate_price": p, "pred_volume": v, "expected_profit": profit})
    results = pd.DataFrame(rows).sort_values("expected_profit", ascending=False).reset_index(drop=True)
    best_raw_price = float(results.loc[0, "candidate_price"])
    best_guarded = apply_guardrails(best_raw_price, today)
    return results, best_guarded

# ---------------------------
# Plot helpers
# ---------------------------

def plot_feature_importances(series: pd.Series, out_file: str):
    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(series))))
    series.head(30).sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Feature importances")
    ax.set_xlabel("Importance")
    save_plot(fig, out_file)

def plot_profit_grid(df_candidates: pd.DataFrame, recommended_price: float, out_file: str):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df_candidates["candidate_price"], df_candidates["expected_profit"], marker="o")
    ax.axvline(recommended_price, linestyle="--", color="red", label="recommended")
    ax.set_xlabel("Candidate price")
    ax.set_ylabel("Expected profit")
    ax.set_title("Expected profit vs candidate price")
    ax.legend()
    save_plot(fig, out_file)

def plot_pred_vs_actual(y_true_train, y_pred_train, y_true_test, y_pred_test, out_file: str):
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    axes[0].scatter(y_true_train, y_pred_train, alpha=0.4)
    axes[0].set_xlabel("Actual (train)"); axes[0].set_ylabel("Predicted (train)")
    axes[0].set_title("Train: predicted vs actual")
    axes[1].scatter(y_true_test, y_pred_test, alpha=0.4)
    axes[1].set_xlabel("Actual (test)"); axes[1].set_ylabel("Predicted (test)")
    axes[1].set_title("Test: predicted vs actual")
    save_plot(fig, out_file)

# ---------------------------
# Main
# ---------------------------

def main_cli():
    ap = argparse.ArgumentParser(description="ML GradientBoosting Fuel Price Optimizer")
    ap.add_argument("--data", default="data/oil_retail_history.xlsx")
    ap.add_argument("--today", default="data/today_example.json")
    ap.add_argument("--out", "--outdir", dest="outdir", default="outputs")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()

    outdir = args.outdir
    ensure_dir(outdir)

    print("Loading historical data...")
    df = load_history(args.data)
    today = load_today(args.today)

    data_summary(df)

    print("Building features...")
    df_feat = build_features(df, with_lags=True)
    print(f"After feature creation: rows={len(df_feat)}, cols={len(df_feat.columns)}")

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
    print("\nCandidate features:", candidate_features)

    X_train = train_df[candidate_features].copy()
    y_train = train_df["volume"].astype(float).copy()
    X_test = test_df[candidate_features].copy()
    y_test = test_df["volume"].astype(float).copy()

    top_k = args.top_k if args.top_k <= len(candidate_features) else len(candidate_features)
    print(f"\nSelecting top {top_k} features by importance...")
    selected = select_top_features_by_importance(X_train, y_train, top_k=top_k)
    print("Selected features:", selected)

    X_train_sel = X_train[selected].copy()
    X_test_sel = X_test[selected].copy()

    print("\nComputing simple time-series CV baseline...")
    tss = TimeSeriesSplit(n_splits=5)
    maes, rmses = [], []
    for train_idx, test_idx in tss.split(X_train_sel):
        m = GradientBoostingRegressor(random_state=42, n_estimators=150)
        m.fit(X_train_sel.iloc[train_idx], y_train.iloc[train_idx])
        pred = m.predict(X_train_sel.iloc[test_idx])
        maes.append(mean_absolute_error(y_train.iloc[test_idx], pred))
        rmses.append(mean_squared_error(y_train.iloc[test_idx], pred) ** 0.5)
    print("CV baseline MAE (mean):", np.mean(maes), "RMSE (mean):", np.mean(rmses))

    print("\nTraining final model...")
    model, meta = train_gbr(X_train_sel, y_train, tune=args.tune)

    print("\nEvaluating model...")
    y_pred_train = model.predict(X_train_sel)
    y_pred_test = model.predict(X_test_sel)
    metrics_train = compute_metrics(y_train.values, y_pred_train)
    metrics_test = compute_metrics(y_test.values, y_pred_test)
    print("TRAIN metrics:", metrics_train)
    print("TEST metrics:", metrics_test)

    # Save pred vs actual plot
    plot_pred_vs_actual(y_train.values, y_pred_train, y_test.values, y_pred_test, os.path.join(outdir, "pred_vs_actual.png"))

    # Feature importances
    feat_importances = pd.Series(model.feature_importances_, index=selected).sort_values(ascending=False)
    plot_feature_importances = lambda s, p: plot_feature_importances  # no-op to avoid overshadow; we'll call function below
    plot_feature_importances = lambda s, p: None  # placeholder to avoid name warning
    # Save using explicit function
    plot_feature_importances(feat_importances, os.path.join(outdir, "feature_importances.png"))  # replaced below

    # Actually call the plotting helper defined earlier:
    plot_feature_importances(feat_importances, os.path.join(outdir, "feature_importances.png"))

    # Sweep grid for today's recommendation
    print("\nEvaluating candidate price grid...")
    df_candidates, recommended_price = evaluate_candidate_grid(model, selected, df, today, step=args.step)
    print("\nTop candidate prices (by expected profit):")
    print(df_candidates.head(10).to_string(index=False))
    print(f"\nRecommended price after guardrails: {recommended_price:.2f}")

    # Save outputs
    save_candidates_to_excel(df_candidates, os.path.join(outdir, "candidate_prices.xlsx"))
    plot_profit_grid(df_candidates, recommended_price, os.path.join(outdir, "profit_vs_price.png"))

    rec = {
        "recommended_price": recommended_price,
        "raw_best_price": float(df_candidates.loc[0,"candidate_price"]),
        "today": {
            "date": str(today["date"].date()),
            "yesterday_price": today["price"],
            "cost": today["cost"],
            "comp_mean": today["comp_mean"],
            "comp1_price": today["comp1_price"],
            "comp2_price": today["comp2_price"],
            "comp3_price": today["comp3_price"]
        },
        "model_metrics": {
            "train": metrics_train,
            "test": metrics_test
        },
        "selected_features": selected,
        "grid_summary": {
            "n_candidates": int(len(df_candidates)),
            "top_candidate_price": float(df_candidates.loc[0,"candidate_price"]),
            "top_expected_profit": float(df_candidates.loc[0,"expected_profit"])
        }
    }
    save_json(rec, os.path.join(outdir, "recommendation.json"))

    # feature importances CSV
    Path(os.path.join(outdir, "feature_importances.csv")).write_text(feat_importances.to_csv())
    print(f"Saved feature importances CSV: {os.path.join(outdir, 'feature_importances.csv')}")

    print("\nAll outputs saved to:", outdir)
    print("Done.")

if __name__ == "__main__":
    main_cli()
