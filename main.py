
"""
Model Benchmarking and Performance Comparison
Regression and Classification Model Evaluation
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage
)

warnings.filterwarnings("ignore")

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20

OUTPUT_DIR = "outputs"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
GRAPH_DIR = os.path.join(OUTPUT_DIR, "graphs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)


# --------------------------------------------------
# LOAD DATASETS
# --------------------------------------------------

def load_datasets():
    """Load regression and classification datasets."""

    diabetes = load_diabetes()
    cancer = load_breast_cancer()

    X_reg = pd.DataFrame(
        diabetes.data,
        columns=diabetes.feature_names
    )
    y_reg = diabetes.target

    X_clf = pd.DataFrame(
        cancer.data,
        columns=cancer.feature_names
    )
    y_clf = cancer.target

    return X_reg, y_reg, X_clf, y_clf


# --------------------------------------------------
# SPLIT DATASETS
# --------------------------------------------------

def split_datasets(X_reg, y_reg, X_clf, y_clf):
    """Split datasets into training and testing sets."""

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
        train_test_split(
            X_reg,
            y_reg,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
    )

    X_clf_train, X_clf_test, y_clf_train, y_clf_test = (
        train_test_split(
            X_clf,
            y_clf,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_clf
        )
    )

    return (
        X_reg_train, X_reg_test, y_reg_train, y_reg_test,
        X_clf_train, X_clf_test, y_clf_train, y_clf_test
    )


# --------------------------------------------------
# DEFINE MODELS
# --------------------------------------------------

def create_regression_models():
    """Create regression model pipelines."""

    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),

        "Support Vector Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf"))
        ])
    }


def create_classification_models():
    """Create classification model pipelines."""

    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=5000,
                random_state=RANDOM_STATE
            ))
        ]),

        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),

        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf"))
        ])
    }


# --------------------------------------------------
# TRAIN REGRESSION MODELS
# --------------------------------------------------

def evaluate_regression_models(models, X_train, X_test, y_train, y_test):
    """Train regression models and calculate performance metrics."""

    results = []
    predictions = {}

    for name, model in models.items():

        print(f"Training: {name}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        predictions[name] = y_pred

        mse = mean_squared_error(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2 Score": r2_score(y_test, y_pred)
        })

        joblib.dump(
            model,
            os.path.join(MODEL_DIR, name.replace(" ", "_") + ".pkl")
        )

    return pd.DataFrame(results), predictions


# --------------------------------------------------
# TRAIN CLASSIFICATION MODELS
# --------------------------------------------------

def evaluate_classification_models(
    models, X_train, X_test, y_train, y_test
):
    """Train classification models and calculate performance metrics."""

    results = []
    predictions = {}

    for name, model in models.items():

        print(f"Training: {name}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        predictions[name] = y_pred

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, y_pred, zero_division=0
            )
        })

        joblib.dump(
            model,
            os.path.join(MODEL_DIR, name.replace(" ", "_") + ".pkl")
        )

    return pd.DataFrame(results), predictions


# --------------------------------------------------
# VISUALIZE REGRESSION RESULTS
# --------------------------------------------------

def create_regression_graph(results):
    """Create regression model comparison graph."""

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=results,
        x="Model",
        y="R2 Score"
    )

    plt.title("Regression Model R2 Score Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        os.path.join(GRAPH_DIR, "regression_comparison.png"),
        dpi=300
    )
    plt.close()


# --------------------------------------------------
# VISUALIZE CLASSIFICATION RESULTS
# --------------------------------------------------

def create_classification_graph(results):
    """Create classification model comparison graph."""

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

    plot_data = results.melt(
        id_vars="Model",
        value_vars=metrics,
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(12, 7))

    sns.barplot(
        data=plot_data,
        x="Model",
        y="Score",
        hue="Metric"
    )

    plt.title("Classification Model Performance Comparison")
    plt.xticks(rotation=15)
    plt.ylim(0, 1.1)
    plt.tight_layout()

    plt.savefig(
        os.path.join(GRAPH_DIR, "classification_comparison.png"),
        dpi=300
    )
    plt.close()


# --------------------------------------------------
# CONFUSION MATRICES
# --------------------------------------------------

def create_confusion_matrices(y_test, predictions):
    """Generate confusion matrices for classification models."""

    for name, y_pred in predictions.items():

        matrix = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.tight_layout()

        filename = name.replace(" ", "_") + "_confusion_matrix.png"

        plt.savefig(
            os.path.join(GRAPH_DIR, filename),
            dpi=300
        )

        plt.close()


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

def save_results(reg_results, clf_results):
    """Save model comparison results as CSV files."""

    reg_results.to_csv(
        os.path.join(OUTPUT_DIR, "regression_results.csv"),
        index=False
    )

    clf_results.to_csv(
        os.path.join(OUTPUT_DIR, "classification_results.csv"),
        index=False
    )


# --------------------------------------------------
# GENERATE PDF REPORT
# --------------------------------------------------

def generate_pdf_report(reg_results, clf_results):
    """Generate a PDF benchmarking report."""

    pdf_path = os.path.join(
        OUTPUT_DIR,
        "model_benchmarking_report.pdf"
    )

    document = SimpleDocTemplate(
        pdf_path,
        pagesize=A4
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(
        "Model Benchmarking and Performance Comparison",
        styles["Title"]
    ))

    story.append(Spacer(1, 20))

    story.append(Paragraph(
        "1. Introduction",
        styles["Heading2"]
    ))

    story.append(Paragraph(
        "This project compares different machine learning algorithms "
        "using regression and classification datasets. Models are "
        "evaluated using standard performance metrics.",
        styles["BodyText"]
    ))

    story.append(Spacer(1, 15))

    story.append(Paragraph(
        "2. Regression Model Results",
        styles["Heading2"]
    ))

    reg_data = [list(reg_results.columns)]

    for row in reg_results.round(4).values.tolist():
        reg_data.append([str(value) for value in row])

    reg_table = Table(reg_data, repeatRows=1)

    reg_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER")
    ]))

    story.append(reg_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph(
        "3. Classification Model Results",
        styles["Heading2"]
    ))

    clf_data = [list(clf_results.columns)]

    for row in clf_results.round(4).values.tolist():
        clf_data.append([str(value) for value in row])

    clf_table = Table(clf_data, repeatRows=1)

    clf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkgreen),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER")
    ]))

    story.append(clf_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph(
        "4. Best Performing Models",
        styles["Heading2"]
    ))

    best_reg = reg_results.sort_values(
        by="R2 Score", ascending=False
    ).iloc[0]

    best_clf = clf_results.sort_values(
        by="F1 Score", ascending=False
    ).iloc[0]

    story.append(Paragraph(
        f"Best Regression Model: {best_reg['Model']} "
        f"(R2 Score: {best_reg['R2 Score']:.4f})",
        styles["BodyText"]
    ))

    story.append(Paragraph(
        f"Best Classification Model: {best_clf['Model']} "
        f"(F1 Score: {best_clf['F1 Score']:.4f})",
        styles["BodyText"]
    ))

    story.append(Spacer(1, 15))

    story.append(Paragraph(
        "5. Performance Graphs",
        styles["Heading2"]
    ))

    for graph in [
        "regression_comparison.png",
        "classification_comparison.png"
    ]:

        graph_path = os.path.join(GRAPH_DIR, graph)

        if os.path.exists(graph_path):
            story.append(PDFImage(graph_path, width=420, height=230))
            story.append(Spacer(1, 10))

    story.append(Paragraph(
        "6. Conclusion",
        styles["Heading2"]
    ))

    story.append(Paragraph(
        "The project successfully trained and compared three regression "
        "and three classification algorithms. Performance metrics and "
        "visualizations help identify suitable models for the selected "
        "datasets. The best model depends on the evaluation metric and "
        "the requirements of the application.",
        styles["BodyText"]
    ))

    document.build(story)

    print(f"\nPDF report generated: {pdf_path}")


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():
    """Execute the complete model benchmarking workflow."""

    print("\nMODEL BENCHMARKING PROJECT")
    print("=" * 45)

    X_reg, y_reg, X_clf, y_clf = load_datasets()

    print("\nREGRESSION DATASET")
    print("Total Samples:", len(X_reg))

    print("\nCLASSIFICATION DATASET")
    print("Total Samples:", len(X_clf))

    (
        X_reg_train, X_reg_test, y_reg_train, y_reg_test,
        X_clf_train, X_clf_test, y_clf_train, y_clf_test
    ) = split_datasets(X_reg, y_reg, X_clf, y_clf)

    print("\nDataset splitting completed.")

    print("\nTRAINING REGRESSION MODELS")
    regression_models = create_regression_models()

    reg_results, reg_predictions = evaluate_regression_models(
        regression_models,
        X_reg_train,
        X_reg_test,
        y_reg_train,
        y_reg_test
    )

    print("\nTRAINING CLASSIFICATION MODELS")
    classification_models = create_classification_models()

    clf_results, clf_predictions = evaluate_classification_models(
        classification_models,
        X_clf_train,
        X_clf_test,
        y_clf_train,
        y_clf_test
    )

    print("\nREGRESSION RESULTS")
    print(reg_results.to_string(index=False))

    print("\nCLASSIFICATION RESULTS")
    print(clf_results.to_string(index=False))

    print("\nGENERATING GRAPHS")
    create_regression_graph(reg_results)
    create_classification_graph(clf_results)
    create_confusion_matrices(y_clf_test, clf_predictions)

    print("\nSAVING RESULTS")
    save_results(reg_results, clf_results)

    print("\nGENERATING PDF REPORT")
    generate_pdf_report(reg_results, clf_results)

    print("\nPROJECT COMPLETED SUCCESSFULLY!")
    print("Check the outputs folder for all generated files.")


if __name__ == "__main__":
    main()
