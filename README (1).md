# Student Performance Analysis and Prediction

An end-to-end data science project: explore a student dataset, find which factors affect the final score, and build models that predict `total_score`.

## Goal

1. Find which factors (study hours, attendance, participation) affect a student's total score.
2. Build and compare models that predict the score.

## Dataset

- File: `student_performance.csv`
- Size: 1,000,000 students, 6 columns, no missing values
- Source: *(add the link or name of where you got the dataset)*

| Column | Meaning |
|---|---|
| `student_id` | Student number (not used) |
| `weekly_self_study_hours` | Self-study hours per week |
| `attendance_percentage` | Attendance in % |
| `class_participation` | Participation score (0 to 10) |
| `total_score` | Total marks (target) |
| `grade` | A, B, C, D or F (made from `total_score`) |

## Tools

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Google Colab

## Method

1. **Data checking:** missing values, duplicates, value ranges, clipping, grade ranges
2. **EDA:** histograms, scatter plots, boxplot, group averages, correlation heatmap
3. **Preparation:** removed `grade` (data leakage), 80/20 train/test split stratified by grade
4. **Modeling:** baseline, Linear Regression, Linear Regression with a squared feature, Random Forest

## Key Findings

- **Study hours is the only factor that matters.** Correlation with score is 0.81. Each extra study hour adds about 2 marks until the score reaches 100.
- **Attendance and participation show no effect.** Correlation is about 0, and average score is the same in every group.
- **About 27% of students have exactly 100 marks.** The scores look capped (clipped) at 100, which flattens the top of the relationship.
- **Grades are imbalanced:** A is about 55% of students, F is only 0.6%.

## Model Results (on 200,000 test students)

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Baseline (average) | 12.92 | 15.44 | 0.000 |
| Linear Regression | 7.18 | 9.02 | 0.659 |
| Linear Regression + squared study hours | 6.33 | 8.30 | 0.711 |
| Random Forest | 6.14 | 8.25 | 0.715 |

Random Forest is the best model. Its predictions never go above 100, while plain Linear Regression predicted scores up to about 130.

## Limitations

- The data looks **synthetic** (computer generated). Real students usually show some effect of attendance and participation, so these findings should not be applied to real schools.
- The cap at 100 hides differences between top students, which limits every model (R2 stays near 0.71).
- Random Forest was trained on a 200,000 row sample for speed.
- `grade` was not used as an input because it is created from `total_score` (data leakage).

## How to Run

1. Open `student_performance_eda.ipynb` in Google Colab.
2. Upload `student_performance.csv` to the Colab files panel.
3. Click Runtime, then Run all.

## Project Files

```
student-performance-analysis/
├── README.md
├── student_performance_eda.ipynb
└── .gitignore
```

## Possible Next Steps

- Try Gradient Boosting models
- Predict `grade` as a classification problem and handle the class imbalance
- Test the same approach on a real student dataset
