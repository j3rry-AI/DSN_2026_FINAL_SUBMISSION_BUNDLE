# DSN Bootcamp Qualification Hackathon 2026  ML Track
## Final Solution Writeup

**Task:** Predict `total_sales` for 1,705 test rows (regression).
**Final validated score:** 10 seed leakage safe nested CV RMSE = **1075.12**
**Submission:** `submission.csv` (1,705 predictions, all format checks passed).



## 1. Understanding the problem
* Train: 6,818 rows x 12 features + target; Test: 1,705 rows.
* Target `total_sales`: mean 2,174.76, std 1,697.89, right skewed (1.15).
* Mix of numeric (price, weight, visibility, store age), high cardinality categoricals
  (`product_code`: 1,555 unique; `store_code`: 10 stores), and low cardinality categoricals.

## 2. Data analysis  factors influencing sales (EDA)
Key documented patterns (see `eda/eda_overview.png`):
* **Price is the dominant driver**: Pearson r = 0.57, Spearman = 0.57 with total_sales.
* **Store effects are large**: mean sales per store vary with CV = 0.49 across stores.
* **Tier effect (non monotonic)**: Tier_2 (2,342) > Tier_3 (2,254) > Tier_1 (1,868)  Tier_1
  being lowest is counterintuitive and worth noting in the report.
* **Store size**: Medium (2,671) > Large (2,255) > Small (1,918) > Unknown (1,829).
* **Category differences**: Starchy Foods (2,408), Household (2,292), Fruits & Vegetables
  (2,278) lead mean sales.
* **Visibility**: weak negative Spearman (negative 0.11)  low visibility items sell less, but the
  effect is small.
* **Missingness**: `store_size` missing 28% of train (structurally  3 whole stores unknown);
  `product_weight_kg` missing 18%. Rows missing weight actually show HIGHER mean sales
  (2,444 vs 2,116), so missingness itself is informative and kept as a flag feature.

## 3. Cleaning / preprocessing
* Normalized categorical labels (case and format duplicates in `product_category`, `fat_content`,
  `store_format`, whitespace in tiers).
* `store_size`: filled with "Unknown" + binary missing flag (structural missingness).
* `product_weight_kg`: median imputation + binary missing flag.
* Price and weight log transforms; price z score (price_log, price_z, weight_log_imp).

## 4. Feature engineering
* **Leakage safe target encoding** (smoothed, k=5) for `product_code` and
  `product_category_clean`  computed out of fold inside every CV fold (see Section 5).
* Validated negative results (dropped): `store_code` target encoding, raw interaction
  features (price x visibility, price x weight, shelf x weight, store_age x price).

## 5. Model & evaluation methodology
* **Model:** LightGBM (regression, RMSE), num_leaves=127, lr=0.05, feature_fraction=0.85,
  bagging_fraction=0.55, min_data_in_leaf=100, lambda_l2=5, early stopping (200).
* **Validation integrity (critical fix):** nested CV. Outer 5 fold CV evaluates the model;
  inside each outer fold, target encodings are rebuilt from that fold's training data only
  (inner 5 fold OOF for train rows; smoothed outer train mapping for validation rows).
  An earlier flat CV pipeline inflated the score to ~1070.98 by leaking validation targets
  into reused encodings; under the corrected protocol the honest score is 1075.12
  (10 seed OOF averaged, seeds 42/7/2024/97/123/3/11/777/314/999).
* Seed averaging alone is worth ~5 RMSE; parameter screens (bagging, leaves, L2, lr,
  feature_fraction) all fell within noise and none were adopted.

## 6. Competitive experiment (evidence gated)
Diverse model test under the same leakage safe protocol:
* CatBoost: not installable in this environment (no network/pip).
* XGBoost: not available either.
* sklearn HistGradientBoosting + TE features (native alternative): 1,171.86 RMSE
  (stronger configuration: 1,212.74)  9 to 13% worse than the 1,075.12 benchmark, far outside the
  ~2% blend worthiness threshold. **Discarded without blending.** LightGBM retained.

## 7. Final prediction
* Retrained on all 6,818 rows; target encodings from full train OOF (5 fold) + smoothed
  full train mappings for test; n_estimators = 82 (mean early stopping iteration);
  10 seed prediction average; clipped to [0, 13000].
* Prediction sanity: mean 2,178 / std 1,256 vs target mean 2,175 / std 1,698; range 165 to 5,953.
* Submission checks: 1,705 rows; columns id/total_sales; IDs match sample_submission in
  order; no NaNs; no duplicates; numeric; not negative. ALL PASS.

## 8. Conclusion
The DSN workflow is complete end to end: understand, analyse, clean, EDA, feature
engineering, model and evaluate, predict, and verify the submission file. The honest, leakage free
validated result is LightGBM + nested TE + 10 seed averaging at RMSE 1075.12.
