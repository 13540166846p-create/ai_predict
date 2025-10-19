import argparse
import json
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from joblib import dump

try:
    from lightgbm import LGBMRegressor
except Exception as e:
    raise RuntimeError("LightGBM is required. Please install lightgbm before running.") from e


def read_table(path: str, sheet_name: str | None = None) -> pd.DataFrame:
    path_lower = path.lower()
    if path_lower.endswith(('.xlsx', '.xls')):
        return pd.read_excel(path, sheet_name=sheet_name)
    return pd.read_csv(path)


def detect_targets(df_a: pd.DataFrame, df_b: pd.DataFrame, provided: List[str] | None) -> List[str]:
    if provided:
        missing = [t for t in provided if t not in df_a.columns or t not in df_b.columns]
        if missing:
            raise ValueError(f"Target columns not found in both datasets: {missing}")
        return provided
    # Heuristic: common columns whose name suggests performance metrics
    candidate_keywords = [
        'gain', 'a0', 'dcgain', 'gbw', 'ugbw', 'pm', 'phase', 'power', 'idd', 'icc', 'noise',
        'vos', 'offset', 'slew', 'sr', 'psrr', 'cmrr', 'thd', 'snr', 'enbw'
    ]
    common = set(df_a.columns).intersection(set(df_b.columns))
    candidates = [c for c in common if any(k in c.lower() for k in candidate_keywords)]
    if not candidates:
        raise ValueError(
            "Could not auto-detect targets. Please pass --target-cols explicitly (space-separated)."
        )
    return sorted(candidates)


def select_feature_columns(
    df_a: pd.DataFrame, df_b: pd.DataFrame, targets: List[str], provided: List[str] | None
) -> List[str]:
    if provided:
        # Validate all provided feature columns exist in at least one dataset
        not_found = [c for c in provided if c not in df_a.columns and c not in df_b.columns]
        if not_found:
            raise ValueError(f"Feature columns not found in datasets: {not_found}")
        return provided
    # Default: all numeric columns except targets
    def numeric_cols(df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    a_num = set(numeric_cols(df_a))
    b_num = set(numeric_cols(df_b))
    feat = list((a_num.union(b_num)) - set(targets))
    if not feat:
        raise ValueError("No numeric feature columns detected. Provide --feature-cols explicitly.")
    return sorted(feat)


def align_feature_frames(
    df: pd.DataFrame, feature_cols: List[str], fill_values: Dict[str, float]
) -> pd.DataFrame:
    # Ensure all required features present; missing columns are filled with 0 then imputed
    present = {c for c in df.columns}
    missing = [c for c in feature_cols if c not in present]
    if missing:
        for c in missing:
            df[c] = 0.0
    X = df[feature_cols].copy()
    # Impute using provided medians (computed from A)
    for c in feature_cols:
        if c in fill_values:
            X[c] = X[c].fillna(fill_values[c])
        else:
            X[c] = X[c].fillna(0.0)
    return X


def compute_feature_medians(df_a: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    med = {}
    for c in feature_cols:
        if c in df_a.columns and pd.api.types.is_numeric_dtype(df_a[c]):
            med[c] = float(df_a[c].median()) if not df_a[c].dropna().empty else 0.0
        else:
            med[c] = 0.0
    return med


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_pred - y_true) / denom)))


def train_and_finetune(
    X_a: pd.DataFrame,
    y_a: pd.DataFrame,
    X_b: pd.DataFrame,
    y_b: pd.DataFrame,
    targets: List[str],
    model_dir: str,
    seed: int,
    n_estimators_pretrain: int,
    n_estimators_finetune: int,
    learning_rate: float,
    num_leaves: int,
    max_depth: int,
    test_size: float,
) -> Dict[str, Dict[str, float]]:
    os.makedirs(model_dir, exist_ok=True)

    metrics: Dict[str, Dict[str, float]] = {}

    Xb_tr, Xb_va, yb_tr_df, yb_va_df = train_test_split(
        X_b, y_b, test_size=test_size, random_state=seed
    )

    for tgt in targets:
        print(f"[INFO] Training target: {tgt}")
        y_a_t = y_a[tgt].values
        yb_tr = yb_tr_df[tgt].values
        yb_va = yb_va_df[tgt].values

        pre_model = LGBMRegressor(
            n_estimators=n_estimators_pretrain,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            min_child_samples=20,
        )
        pre_model.fit(X_a, y_a_t)

        ft_model = LGBMRegressor(
            n_estimators=n_estimators_finetune,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            min_child_samples=10,
        )

        ft_model.fit(
            Xb_tr,
            yb_tr,
            eval_set=[(Xb_va, yb_va)],
            eval_metric='l2',
            early_stopping_rounds=100,
            init_model=pre_model.booster_,
            verbose=False,
        )

        va_pred = ft_model.predict(Xb_va)
        tgt_metrics = {
            'rmse': rmse(yb_va, va_pred),
            'mae': float(mean_absolute_error(yb_va, va_pred)),
            'mape': mape(yb_va, va_pred),
            'r2': float(r2_score(yb_va, va_pred)),
            'best_iteration': int(getattr(ft_model, 'best_iteration_', ft_model.n_estimators)),
        }
        metrics[tgt] = tgt_metrics
        print(f"[METRICS][{tgt}] RMSE={tgt_metrics['rmse']:.4f} MAE={tgt_metrics['mae']:.4f} MAPE={tgt_metrics['mape']:.4f} R2={tgt_metrics['r2']:.4f}")

        out_path = os.path.join(model_dir, f"{tgt}_lgbm.pkl")
        dump(ft_model, out_path)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Two-stage LightGBM training: A pretrain -> B finetune")
    parser.add_argument('--a_path', required=True, help='Path to A-process dataset (xlsx/csv)')
    parser.add_argument('--b_path', required=True, help='Path to B-process dataset (xlsx/csv)')
    parser.add_argument('--a_sheet', default=None, help='Excel sheet name for A (optional)')
    parser.add_argument('--b_sheet', default=None, help='Excel sheet name for B (optional)')
    parser.add_argument('--target-cols', nargs='+', default=None, help='Target column names (space-separated)')
    parser.add_argument('--feature-cols', nargs='+', default=None, help='Optional feature column names (space-separated)')
    parser.add_argument('--model-dir', default='models_lightgbm', help='Directory to save models and artifacts')
    parser.add_argument('--test-size', type=float, default=0.2, help='Validation split ratio on B-process data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-estimators-pretrain', type=int, default=800)
    parser.add_argument('--n-estimators-finetune', type=int, default=300)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--num-leaves', type=int, default=64)
    parser.add_argument('--max-depth', type=int, default=-1)

    args = parser.parse_args()

    print("[INFO] Loading datasets...")
    df_a = read_table(args.a_path, sheet_name=args.a_sheet)
    df_b = read_table(args.b_path, sheet_name=args.b_sheet)

    print(f"[INFO] A shape: {df_a.shape} | B shape: {df_b.shape}")

    targets = detect_targets(df_a, df_b, args.target_cols)
    print(f"[INFO] Targets: {targets}")

    features = select_feature_columns(df_a, df_b, targets, args.feature_cols)
    print(f"[INFO] Num features: {len(features)}")

    # Prepare medians from A for imputation
    medians = compute_feature_medians(df_a, features)

    # Align and impute features
    X_a = align_feature_frames(df_a, features, medians)
    X_b = align_feature_frames(df_b, features, medians)

    # Targets
    y_a = df_a[targets].copy()
    y_b = df_b[targets].copy()

    print("[INFO] Training...")
    metrics = train_and_finetune(
        X_a=X_a,
        y_a=y_a,
        X_b=X_b,
        y_b=y_b,
        targets=targets,
        model_dir=args.model_dir,
        seed=args.seed,
        n_estimators_pretrain=args.n_estimators_pretrain,
        n_estimators_finetune=args.n_estimators_finetune,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        test_size=args.test_size,
    )

    # Save artifacts
    os.makedirs(args.model_dir, exist_ok=True)
    with open(os.path.join(args.model_dir, 'features.json'), 'w', encoding='utf-8') as f:
        json.dump(features, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.model_dir, 'targets.json'), 'w', encoding='utf-8') as f:
        json.dump(targets, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.model_dir, 'feature_medians.json'), 'w', encoding='utf-8') as f:
        json.dump(medians, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.model_dir, 'metrics_valid_b.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Models and artifacts saved to: {args.model_dir}")


if __name__ == '__main__':
    main()
