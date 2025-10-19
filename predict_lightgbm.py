import argparse
import json
import os
from typing import List, Dict

import numpy as np
import pandas as pd
from joblib import load


def read_table(path: str, sheet_name: str | None = None) -> pd.DataFrame:
    path_lower = path.lower()
    if path_lower.endswith(('.xlsx', '.xls')):
        return pd.read_excel(path, sheet_name=sheet_name)
    return pd.read_csv(path)


def align_feature_frames(
    df: pd.DataFrame, feature_cols: List[str], fill_values: Dict[str, float]
) -> pd.DataFrame:
    present = {c for c in df.columns}
    missing = [c for c in feature_cols if c not in present]
    if missing:
        for c in missing:
            df[c] = 0.0
    X = df[feature_cols].copy()
    for c in feature_cols:
        X[c] = X[c].fillna(fill_values.get(c, 0.0))
    return X


def main():
    parser = argparse.ArgumentParser(description="Predict using LightGBM models")
    parser.add_argument('--input', required=True, help='Path to input dataset (xlsx/csv)')
    parser.add_argument('--sheet', default=None, help='Excel sheet name (optional)')
    parser.add_argument('--model-dir', default='models_lightgbm', help='Directory with saved models')
    parser.add_argument('--output', default='predictions.csv', help='Output CSV path for predictions')

    args = parser.parse_args()

    with open(os.path.join(args.model_dir, 'features.json'), 'r', encoding='utf-8') as f:
        features = json.load(f)
    with open(os.path.join(args.model_dir, 'targets.json'), 'r', encoding='utf-8') as f:
        targets = json.load(f)
    with open(os.path.join(args.model_dir, 'feature_medians.json'), 'r', encoding='utf-8') as f:
        medians = json.load(f)

    models = {}
    for tgt in targets:
        model_path = os.path.join(args.model_dir, f"{tgt}_lgbm.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found for target '{tgt}': {model_path}")
        models[tgt] = load(model_path)

    df = read_table(args.input, sheet_name=args.sheet)
    X = align_feature_frames(df, features, medians)

    pred_df = pd.DataFrame(index=df.index)
    for tgt, model in models.items():
        pred = model.predict(X)
        pred_df[tgt] = pred

    pred_df.to_csv(args.output, index=False)
    print(f"[DONE] Saved predictions to {args.output}")


if __name__ == '__main__':
    main()
