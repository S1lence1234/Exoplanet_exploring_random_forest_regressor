import time
import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier, Pool
from IPython.display import display

SEED        = 65
N_SPLITS    = 5
LR          = 0.001
DEPTH       = 10
ITERATIONS  = 1000
ES_ROUNDS   = 100
L2_REG      = 1.0
RUN_NAME    = "catboost_cv"

# Укажите путь к вашему train-файлу:
TRAIN_PATH = "train.csv" 

def set_seed(seed=SEED):
    import random, os
    random.seed(seed)
    np.random.seed(seed)

set_seed()

def detect_columns(df: pd.DataFrame):
    """Определяем имя таргета и id-колонки, если есть."""
    target_candidates = ["target", "type", "label", "y"]
    id_candidates     = ["object_id", "idx", "id"]
    target_col = next((c for c in target_candidates if c in df.columns), None)
    id_col     = next((c for c in id_candidates if c in df.columns), None)
    if target_col is None:
        raise ValueError("Не найдена колонка с таргетом (ожидались: target/type/label/y).")
    return target_col, id_col

def get_feature_lists(df: pd.DataFrame, target_col: str, id_col: Optional[str]):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    for col in [target_col, id_col]:
        if col in num_cols: num_cols.remove(col)
        if col in cat_cols: cat_cols.remove(col)
    return num_cols, cat_cols

df = pd.read_csv(TRAIN_PATH)
target_col, id_col = detect_columns(df) # Загружаем данные и автоматически определяем структуру.

num_cols, cat_cols = get_feature_lists(df, target_col, id_col)
for c in cat_cols:
    df[c] = df[c].astype(str)


# Разделяем фичи/цель
feature_cols = [c for c in df.columns if c not in {target_col, id_col}]
X = df[feature_cols].copy()
y = df[target_col].copy()
cat_idx = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_pred = np.empty(len(X), dtype=object)
fold_models: List[CatBoostClassifier] = []
fold_scores = []
start_time = time.time()

params = dict(
    loss_function="MultiClass",
    eval_metric="MultiClass",          # Macro F1 посчитаем сами
    learning_rate=LR,
    depth=DEPTH,
    l2_leaf_reg=L2_REG,
    random_strength=0.8,
    bagging_temperature=0.2,
    auto_class_weights="Balanced",     # балансируем редкие классы
    iterations=ITERATIONS,
    early_stopping_rounds=ES_ROUNDS,
    allow_writing_files=False,
    thread_count=-1,
    random_state=SEED,
    verbose=False
)

for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
# Для каждого фолда разделяем данные на тренировочную и валидационную части.
    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
    valid_pool = Pool(X_val, y_val, cat_features=cat_idx)
# Создаём специальные объекты `Pool` для CatBoost с указанием категориальных признаков.
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
# 1. Создаём модель с нашими параметрами
# 2. Обучаем её с валидацией и ранней остановкой
    y_val_pred = model.predict(valid_pool).astype(object).ravel()
    score = f1_score(y_val, y_val_pred, average="macro")
    fold_scores.append(score)
    oof_pred[val_idx] = y_val_pred
    fold_models.append(model)
# 1. Делаем предсказания на валидационной части
# 2. Вычисляем F1-score (метрику качества)
# 3. Сохраняем результаты для общей оценки
    print(f"Fold {fold}: Macro F1 = {score:.5f} | best_iter={model.get_best_iteration()}")

oof_macro_f1 = f1_score(y, oof_pred, average="macro")
elapsed = time.time() - start_time
print(f"\nOOF Macro F1: {oof_macro_f1:.5f}  (folds: {[f'{s:.4f}' for s in fold_scores]})")
print(f"Время обучения: {elapsed:.1f} c")
# Вычисляем общую оценку качества и выводим результаты.
# Это как честная оценка — каждый объект предсказывается моделью, которая его не видела при обучении.

# Подробный отчёт по каждому классу (precision, recall, F1-score).
print("\nОтчёт по классам (OOF):")
print(classification_report(y, oof_pred, digits=4))

# Матрица ошибок — показывает, какие классы путает модель.:
labels_order = sorted(y.unique().tolist())
cm = pd.DataFrame(confusion_matrix(y, oof_pred, labels=labels_order),
                  index=[f"true_{c}" for c in labels_order],
                  columns=[f"pred_{c}" for c in labels_order])
print("\nConfusion matrix (OOF):")
display(cm)

#Новые признаки
X.loc[:, 'g_r'] = X['g_mag'] - X['r_mag']
X.loc[:, 'u_g'] = X['u_mag'] - X['g_mag']
X.loc[:, 'r_i'] = X['r_mag'] - X['i_mag']

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_pred = np.empty(len(X), dtype=object)
fold_models: List[CatBoostClassifier] = []
fold_scores = []
start_time = time.time()

for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
# Для каждого фолда разделяем данные на тренировочную и валидационную части.
    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
    valid_pool = Pool(X_val, y_val, cat_features=cat_idx)
# Создаём специальные объекты `Pool` для CatBoost с указанием категориальных признаков.
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
# 1. Создаём модель с нашими параметрами
# 2. Обучаем её с валидацией и ранней остановкой
    y_val_pred = model.predict(valid_pool).astype(object).ravel()
    score = f1_score(y_val, y_val_pred, average="macro")
    fold_scores.append(score)
    oof_pred[val_idx] = y_val_pred
    fold_models.append(model)
# 1. Делаем предсказания на валидационной части
# 2. Вычисляем F1-score (метрику качества)
# 3. Сохраняем результаты для общей оценки
    print(f"Fold {fold}: Macro F1 = {score:.5f} | best_iter={model.get_best_iteration()}")

oof_macro_f1 = f1_score(y, oof_pred, average="macro")
elapsed = time.time() - start_time
print(f"\nOOF Macro F1: {oof_macro_f1:.5f}  (folds: {[f'{s:.4f}' for s in fold_scores]})")
print(f"Время обучения: {elapsed:.1f} c")
# Вычисляем общую оценку качества и выводим результаты.
# Это как честная оценка — каждый объект предсказывается моделью, которая его не видела при обучении.

# Подробный отчёт по каждому классу (precision, recall, F1-score).
print("\nОтчёт по классам (OOF):")
print(classification_report(y, oof_pred, digits=4))

# Матрица ошибок — показывает, какие классы путает модель.:
labels_order = sorted(y.unique().tolist())
cm = pd.DataFrame(confusion_matrix(y, oof_pred, labels=labels_order),
                  index=[f"true_{c}" for c in labels_order],
                  columns=[f"pred_{c}" for c in labels_order])
print("\nConfusion matrix (OOF):")
display(cm)

# best_iters = [m.get_best_iteration() or ITERATIONS for m in fold_models]
# final_iters = int(np.clip(np.mean(best_iters), 200, ITERATIONS))
# final_params = {**params, "iterations": final_iters, "early_stopping_rounds": None, "verbose": False}

# final_model = CatBoostClassifier(**final_params)
# final_pool  = Pool(X, y, cat_features=cat_idx)
# final_model.fit(final_pool)

# final_model.save_model(f"{RUN_NAME}_model.cbm")
# pd.Series(final_model.get_feature_importance(prettified=False), index=feature_cols)\
#   .sort_values(ascending=False)\
#   .head(25)\
#   .to_csv(f"{RUN_NAME}_top_features.csv")

# print(f"\nФинальная модель обучена на всём train. iterations={final_iters}.")
# print(f"Сохранено: {RUN_NAME }_model.cbm и {RUN_NAME}_top_features.csv")
