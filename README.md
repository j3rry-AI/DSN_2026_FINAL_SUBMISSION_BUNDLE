# DSN Bootcamp Qualification Hackathon 2026

This repository contains the final machine learning submission for the DSN Bootcamp Qualification Hackathon 2026 (ML Track).

The goal is to predict `total_sales` for the test set using a robust, leakage-safe regression pipeline. The final approach uses LightGBM with nested target encoding, missingness-aware preprocessing, and 10-seed model averaging.

## Project summary

- Task: regression on sales prediction
- Dataset: 6,818 training rows and 1,705 test rows
- Target: `total_sales`
- Final validated RMSE: **1075.12**
- Submission file: `submission.csv`

## Key modeling choices

- Leakage-safe nested target encoding for categorical features
- Missing-value flags for structurally missing fields
- Log and z-score transformations for skewed / high-variance numeric predictors
- LightGBM regressor with regularized training setup
- 10-seed averaging to stabilize predictions
- Submission validation checks for schema, nulls, ordering, and numeric integrity

## Repository contents

- `solution_pipeline.py` – reproducible training and prediction pipeline
- `solution_notebook.ipynb` – notebook version of the workflow
- `solution_writeup.md` – detailed writeup of the analysis and modeling decisions
- `submission.csv` – final competition-ready submission
- `oof_predictions.csv` – out-of-fold predictions used for analysis
- `cv_scores.json` – recorded cross-validation scores
- `eda/eda_overview.png` – exploratory data analysis summary
- `model/final_lgbm_10seeds.pkl` – saved final model artifact

## Reproduction

Place the competition files `train.csv`, `test.csv`, and `sample_submission.csv` in a sibling `data/` folder beside `solution_pipeline.py`, then run:

```bash
python solution_pipeline.py
```

The script trains the model, generates predictions, and writes the update to `submission.csv` in the same directory.

## Validation notes

The pipeline enforces a leakage-safe validation protocol:

- target encoding is rebuilt only from each fold's training split
- validation rows only receive encodings learned from the training portion
- the final score is reported under a fair, honest nested-CV evaluation setup

## Result

The final solution delivers a validated RMSE of **1075.12** and the corresponding submission file is included in this repository.

## License

No explicit license has been added for this project bundle.
