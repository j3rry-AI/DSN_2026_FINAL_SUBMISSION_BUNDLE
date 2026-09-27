DSN Bootcamp Qualification Hackathon 2026 ML Track
Final submission bundle

Contents

submission.csv
Final predictions for the competition platform.

solution_pipeline.py
Reproducible LightGBM training and prediction pipeline. It uses leakage safe nested target encoding and ten seed averaging.

solution_notebook.ipynb
Notebook version of the solution workflow.

oof_predictions.csv
Out of fold predictions used for validation analysis.

cv_scores.json
Recorded cross validation scores, including the final averaged RMSE of 1075.12.

eda/eda_overview.png
EDA summary figure.

model/final_lgbm_10seeds.pkl
Saved final LightGBM model artifact.

Running the pipeline

Place train.csv, test.csv, and sample_submission.csv in a folder named data beside solution_pipeline.py. Then run:

python solution_pipeline.py

The script writes submission.csv in the same folder.

The competition upload uses submission.csv. The other files document and reproduce the modelling work.
