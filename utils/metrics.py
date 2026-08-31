import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


def calculate_metrics(
    labels,
    predictions,
    probabilities,
    class_names,
    save_dir="results"
):

    os.makedirs(save_dir, exist_ok=True)

    # ----------------------------------------------------
    # Basic Metrics
    # ----------------------------------------------------

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        pos_label=0
    )

    recall = recall_score(
        labels,
        predictions,
        pos_label=0
    )

    f1 = f1_score(
        labels,
        predictions,
        pos_label=0
    )

    # ----------------------------------------------------
    # Confusion Matrix
    # ----------------------------------------------------

    cm = confusion_matrix(
        labels,
        predictions
    )

    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    # ----------------------------------------------------
    # ROC-AUC
    #
    # Convert COVID (0) -> Positive class (1)
    # Healthy (1) -> Negative class (0)
    # ----------------------------------------------------

    covid_labels = (labels == 0).astype(int)

    auc = roc_auc_score(
        covid_labels,
        probabilities
    )

    # ----------------------------------------------------
    # Classification Report
    # ----------------------------------------------------

    report = classification_report(
        labels,
        predictions,
        target_names=class_names
    )

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"Accuracy      : {accuracy*100:.2f}%")
    print(f"Precision     : {precision*100:.2f}%")
    print(f"Recall        : {recall*100:.2f}%")
    print(f"Sensitivity   : {sensitivity*100:.2f}%")
    print(f"Specificity   : {specificity*100:.2f}%")
    print(f"F1 Score      : {f1*100:.2f}%")
    print(f"ROC AUC       : {auc:.4f}")

    print()
    print(report)

    # ----------------------------------------------------
    # Save CSV
    # ----------------------------------------------------

    dataframe = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "Sensitivity",
            "Specificity",
            "F1 Score",
            "ROC AUC"
        ],

        "Value": [
            accuracy,
            precision,
            recall,
            sensitivity,
            specificity,
            f1,
            auc
        ]

    })

    dataframe.to_csv(

        os.path.join(
            save_dir,
            "metrics.csv"
        ),

        index=False

    )

    # ----------------------------------------------------
    # Save Classification Report
    # ----------------------------------------------------

    with open(

        os.path.join(
            save_dir,
            "classification_report.txt"
        ),

        "w"

    ) as file:

        file.write(report)

    # ----------------------------------------------------
    # Confusion Matrix
    # ----------------------------------------------------

    disp = ConfusionMatrixDisplay(

        confusion_matrix=cm,
        display_labels=class_names

    )

    disp.plot()

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            save_dir,
            "confusion_matrix.png"
        )

    )

    plt.close()

    # ----------------------------------------------------
    # ROC Curve
    # ----------------------------------------------------

    fpr, tpr, _ = roc_curve(

        covid_labels,
        probabilities

    )

    plt.figure(figsize=(6, 6))

    plt.plot(

        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {auc:.4f}"

    )

    plt.plot(

        [0, 1],
        [0, 1],
        linestyle="--"

    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            save_dir,
            "roc_curve.png"
        )

    )

    plt.close()

    # ----------------------------------------------------
    # Precision-Recall Curve
    # ----------------------------------------------------

    precision_curve, recall_curve, _ = precision_recall_curve(

        covid_labels,
        probabilities

    )

    plt.figure(figsize=(6, 6))

    plt.plot(

        recall_curve,
        precision_curve,
        linewidth=2

    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            save_dir,
            "precision_recall_curve.png"
        )

    )

    plt.close()