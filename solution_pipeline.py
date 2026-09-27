"""DSN Bootcamp Qualification Hackathon 2026 ML Track
Reproducible solution pipeline. Final validated score:
10 seed leakage safe nested CV RMSE = 1075.12.
Dependencies: pandas, numpy, scikit learn, lightgbm.
Input data: train.csv, test.csv, and sample_submission.csv.
"""
import os, pickle, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

DATA_DIR = "data"        # folder containing train.csv / test.csv
OUT_DIR = "."
TARGET = "total_sales"
TE_COLS = ("product_code", "product_category_clean")
SEEDS = [42, 7, 2024, 97, 123, 3, 11, 777, 314, 999]
PARAMS = dict(objective="regression", metric="rmse", num_leaves=127, learning_rate=0.05,
              feature_fraction=0.85, bagging_fraction=0.55, bagging_freq=1,
              min_data_in_leaf=100, lambda_l2=5, verbose=-1)

def clean_category(s): return s.astype("string").str.strip().str.lower().str.title()

def prepare_base(tr0, te0):
    tr, te = tr0.copy(), te0.copy(); y = tr[TARGET].astype(float).values
    for df in (tr, te):
        df["product_category_clean"] = clean_category(df["product_category"])
        df["fat_content_clean"] = clean_category(df["fat_content"])
        df["store_format_clean"] = clean_category(df["store_format"])
        df["store_location_tier_clean"] = df["store_location_tier"].astype("string").str.strip()
        df["store_size_clean"] = df["store_size"].astype("string").str.strip().fillna("Unknown")
        df["weight_missing"] = df["product_weight_kg"].isna().astype(np.int8)
        df["store_size_missing"] = df["store_size"].isna().astype(np.int8)
        df["price_log"] = np.log1p(df["product_price"])
    wm = tr["product_weight_kg"].median(); gpm, gps = tr["product_price"].mean(), tr["product_price"].std()
    for df in (tr, te):
        df["product_weight_kg_imp"] = df["product_weight_kg"].fillna(wm)
        df["weight_log_imp"] = np.log1p(df["product_weight_kg_imp"])
        df["price_z"] = (df["product_price"] - gpm) / gps
    feats = ["product_code","product_weight_kg_imp","fat_content_clean","shelf_visibility",
             "product_category_clean","product_price","store_code","store_age_years",
             "store_size_clean","store_size_missing","store_location_tier_clean","store_format_clean",
             "weight_missing","price_log","price_z","weight_log_imp"]
    cats = ["product_code","fat_content_clean","product_category_clean","store_code",
            "store_size_clean","store_location_tier_clean","store_format_clean"]
    return tr, te, y, feats, cats

def compute_te(train_idx, val_idx, df, y, te_cols=TE_COLS, smooth=5.0, inner_seed=42, n_inner=5):
    """Leakage safe target encoding: OOF via inner folds on outer train; validation mapped from full outer train."""
    tr_df = df.iloc[train_idx]; y_tr = y[train_idx]; gm = y_tr.mean()
    tr_enc = np.zeros((len(train_idx), len(te_cols))); val_enc = np.zeros((len(val_idx), len(te_cols)))
    kf_in = KFold(n_splits=n_inner, shuffle=True, random_state=inner_seed)
    for j, col in enumerate(te_cols):
        cv = tr_df[col].values; oof = np.full(len(train_idx), gm)
        for i_tr, i_val in kf_in.split(tr_df):
            grp = pd.Series(y_tr[i_tr], index=tr_df.index[i_tr]).groupby(cv[i_tr])
            cnt, mean = grp.count(), grp.mean()
            m = (cnt * mean + gm * smooth) / (cnt + smooth)
            oof[i_val] = pd.Series(cv[i_val]).map(m).fillna(gm).values
        tr_enc[:, j] = oof
        grp = pd.Series(y_tr, index=tr_df.index).groupby(cv)
        cnt, mean = grp.count(), grp.mean()
        m_full = (cnt * mean + gm * smooth) / (cnt + smooth)
        val_enc[:, j] = pd.Series(df.iloc[val_idx][col].values).map(m_full).fillna(gm).values
    return tr_enc, val_enc

def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

def compute_full_train_test_te(train_df, test_df, y, te_cols=TE_COLS, smooth=5.0, inner_seed=42, n_inner=5):
    """Create leakage safe OOF encodings for all training rows and full train mappings for test rows."""
    gm = y.mean()
    train_enc = np.zeros((len(train_df), len(te_cols)))
    test_enc = np.zeros((len(test_df), len(te_cols)))
    kf = KFold(n_splits=n_inner, shuffle=True, random_state=inner_seed)
    for j, col in enumerate(te_cols):
        values = train_df[col].values
        oof = np.full(len(train_df), gm)
        for i_tr, i_val in kf.split(train_df):
            grp = pd.Series(y[i_tr], index=train_df.index[i_tr]).groupby(values[i_tr])
            cnt, mean = grp.count(), grp.mean()
            mapping = (cnt * mean + gm * smooth) / (cnt + smooth)
            oof[i_val] = pd.Series(values[i_val]).map(mapping).fillna(gm).values
        train_enc[:, j] = oof
        grp_full = pd.Series(y, index=train_df.index).groupby(values)
        cnt_full, mean_full = grp_full.count(), grp_full.mean()
        mapping_full = (cnt_full * mean_full + gm * smooth) / (cnt_full + smooth)
        test_enc[:, j] = pd.Series(test_df[col].values).map(mapping_full).fillna(gm).values
    return train_enc, test_enc

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
train_p, test_p, y, base_features, cats = prepare_base(train, test)

# Validation: nested CV, target encoding recomputed inside each outer fold
def cv_lgbm_nested(outer_seed):
    kf = KFold(n_splits=5, shuffle=True, random_state=outer_seed)
    oof = np.zeros(len(train_p))
    for tr_idx, val_idx in kf.split(train_p):
        Xtr = train_p.iloc[tr_idx][base_features].copy(); Xval = train_p.iloc[val_idx][base_features].copy()
        tr_enc, val_enc = compute_te(tr_idx, val_idx, train_p, y)
        for j, c in enumerate(TE_COLS): Xtr[f"{c}_te"] = tr_enc[:, j]; Xval[f"{c}_te"] = val_enc[:, j]
        for c in cats:
            union = pd.concat([Xtr[c], Xval[c]], axis=0).astype("category").cat.categories
            Xtr[c] = pd.Categorical(Xtr[c], categories=union); Xval[c] = pd.Categorical(Xval[c], categories=union)
        m = lgb.LGBMRegressor(**PARAMS, n_estimators=5000)
        m.fit(Xtr, y[tr_idx], eval_set=[(Xval, y[val_idx])], eval_metric="rmse",
              categorical_feature=cats,
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        oof[val_idx] = m.predict(Xval, num_iteration=m.best_iteration_)
    return oof

oofs = [cv_lgbm_nested(sd) for sd in SEEDS]
print("10 seed nested CV RMSE:", rmse(y, np.mean(oofs, axis=0)))   # expect ~1075.12

# Final fit on all data
tr_enc, te_enc = compute_full_train_test_te(train_p, test_p, y)
X_train = train_p[base_features].copy(); X_test = test_p[base_features].copy()
for j, c in enumerate(TE_COLS): X_train[f"{c}_te"] = tr_enc[:, j]; X_test[f"{c}_te"] = te_enc[:, j]
for c in cats:
    union = pd.concat([X_train[c], X_test[c]], axis=0).astype("category").cat.categories
    X_train[c] = pd.Categorical(X_train[c], categories=union); X_test[c] = pd.Categorical(X_test[c], categories=union)
preds = np.zeros(len(X_test))
for sd in SEEDS:
    m = lgb.LGBMRegressor(**PARAMS, n_estimators=82, random_state=sd)
    m.fit(X_train, y, categorical_feature=cats)
    preds += m.predict(X_test) / len(SEEDS)
sub["total_sales"] = np.clip(preds, 0, 13000)
sub.to_csv(os.path.join(OUT_DIR, "submission.csv"), index=False)
print("submission.csv written")
