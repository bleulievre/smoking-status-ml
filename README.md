# Binary prediction of smoker status from health measurements

Machine Learning final project, MS Business Analytics, Rensselaer Polytechnic Institute (spring 2024).
Python (pandas, scikit-learn, XGBoost, LightGBM), run in Google Colab.

## Question

Can routine health measurements predict whether a person smokes, and which measurements carry the most
information? The competition only scores prediction, so the second part is my own addition: I wanted to see
how smoking, as an environmental exposure, shows up across body measurements and blood markers.

## Data

[Kaggle Playground Series S3E24](https://www.kaggle.com/competitions/playground-series-s3e24), *Binary
Prediction of Smoker Status using Bio-Signals*: a public, synthetic dataset of 159,255 records with 22
measurements (age, height, weight, waist, eyesight, hearing, blood pressure, fasting glucose, lipids,
hemoglobin, urine protein, serum creatinine, liver enzymes, dental caries) and smoking status. The data are
not included here; download `train.csv` from the competition page to run the notebook.

## What the notebook does

[`smoking_status_prediction.ipynb`](smoking_status_prediction.ipynb)
1. **Exploration.** Distributions of age, height and weight, class balance, and a correlation heatmap.
2. **Class imbalance.** Non-smokers outnumber smokers, so the baseline random forest uses balanced class
   weights.
3. **Tuning.** Grid search with 3-fold cross-validation over the number of trees, depth and split size.
4. **Other models.** Logistic regression, gradient boosting, XGBoost and LightGBM, some on reduced samples to
   keep run time manageable.
5. **Interpretation.** Random-forest feature importance, and a test of whether treating height as a
   categorical variable changes performance.

## Results

| Model | Data | Result |
| --- | --- | --- |
| Random forest, balanced class weights | full data, 30% held out | accuracy 0.77, ROC AUC 0.86 |
| Logistic regression (scaled features) | full data, 30% held out | ROC AUC 0.82 |
| Logistic regression / gradient boosting | 1% sample | accuracy 0.69 / 0.72 |
| Random forest / logistic regression / XGBoost / LightGBM | reduced sample | ROC AUC 0.81 / 0.83 / 0.84 / 0.85 |
| Random forest with height in five categories | full data, 30% held out | accuracy 0.69, ROC AUC 0.71 (from the course report; this cell's output is not saved) |

- Best grid-search settings: 150 trees, maximum depth 20, minimum split size 5.
- Kaggle leaderboard scores for my submission: 0.832–0.836 (public and private leaderboards).
- Height, hemoglobin, γ-GTP and triglycerides were the most important features. Weight and waist
  circumference mattered less than several blood markers.
- Turning height into categories lowered both accuracy and AUC, so height carries more information as a
  continuous variable.

## Notes

The data are synthetic, generated from a model trained on a real health-screening dataset, so the patterns
are useful for practising methods but should not be read as findings about real people.

The notebook's section headings and explanatory notes were tidied for readability. The code and its outputs
are as originally run.

## Contact

Yunqi (Vicky) Liu — vickyliu1998424@gmail.com
