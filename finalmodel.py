from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from typing import Optional

model = CatBoostClassifier()

model.load_model('catboost_cv_model.cbm')

def detect_columns(df: pd.DataFrame):
    """Определяем имя таргета и id-колонки, если есть."""
    target_candidates = ["target", "type", "label", "y"]
    id_candidates     = ["object_id", "idx", "id"]
    target_col = next((c for c in target_candidates if c in df.columns), None)
    id_col     = next((c for c in id_candidates if c in df.columns), None)
    return target_col, id_col

def get_feature_lists(df: pd.DataFrame, target_col: str, id_col: Optional[str]):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    for col in [target_col, id_col]:
        if col in num_cols: num_cols.remove(col)
        if col in cat_cols: cat_cols.remove(col)
    return num_cols, cat_cols

df = pd.read_csv('test.csv')
target_col, id_col = detect_columns(df)

num_cols, cat_cols = get_feature_lists(df, target_col, id_col)
for c in cat_cols:
    df[c] = df[c].astype("category")

# Разделяем фичи/цель
feature_cols = [c for c in df.columns if c not in {target_col, id_col}]
X = df[feature_cols].copy()

result = model.predict(X)

print(result)
idx = df[id_col]
sample = pd.read_csv('./sample_submission.csv')
samplecol = sample.columns.tolist()
samplecol.remove('idx')

resdf = pd.DataFrame(columns=samplecol, index=idx)

# next(n for n in samplecol if n == x.name)
def transform(x: pd.Series):
    for id, n in enumerate(result, 0):
        if n == x.name:
            x.loc[idx[id]] = 1
        else:
            x.loc[idx[id]] = 0

resdf.apply(transform)

print(resdf)

resdf.to_csv('submission.csv')

