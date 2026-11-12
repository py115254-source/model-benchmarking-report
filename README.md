
# Model Benchmarking and Performance Comparison

## Project Overview

This project compares six machine learning algorithms using regression and classification datasets. It evaluates their performance using standard machine learning metrics.

## Datasets

### Regression Dataset
- Dataset: Diabetes
- Total Samples: 442
- Training Samples: 353
- Testing Samples: 89

### Classification Dataset
- Dataset: Breast Cancer
- Total Samples: 569
- Training Samples: 455
- Testing Samples: 114

## Machine Learning Models

### Regression
1. Linear Regression
2. Ridge Regression
3. Support Vector Regression

### Classification
1. Logistic Regression
2. K-Nearest Neighbors
3. Support Vector Machine

## Evaluation Metrics

### Regression Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R2 Score

### Classification Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- ReportLab
- Joblib

## Project Structure

```text
model-benchmarking-report/
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
└── outputs/
    ├── regression_results.csv
    ├── classification_results.csv
    ├── model_benchmarking_report.pdf
    ├── models/
    └── graphs/
```

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python main.py
```

## Results

The current execution produced the following results:

- Best Regression Model: Ridge Regression
- Best Regression R2 Score: 0.4541
- Best Classification Models: Logistic Regression and SVM
- Classification Accuracy: 98.25%

## Generated Outputs

The project automatically generates:

- Regression comparison CSV
- Classification comparison CSV
- Performance comparison graphs
- Confusion matrices
- Saved machine learning models
- PDF benchmarking report

## Conclusion

This project demonstrates the training, evaluation, and comparison of multiple machine learning algorithms. It uses numerical metrics and visualizations to understand model performance and identify suitable algorithms for different prediction tasks.
