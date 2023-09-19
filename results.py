import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, average_precision_score
import plotly.graph_objects as go


def plot_graphs(scores, labels, chosen_k, output_dir, test_stage=False, dataset_name="train"):
    path = output_dir + "/train" if not test_stage else output_dir + "/test"
    os.makedirs(path, exist_ok=True)
    plot_scores_histograms(scores.tolist(), labels.tolist(), chosen_k, path, dataset_name)
    ap = plot_precision_recall_curve(labels, scores, chosen_k, path, dataset_name)
    auc = plot_roc_curve(labels, scores, chosen_k, path, dataset_name)
    print(f"The auc for k {chosen_k} is {auc} and the ap is {ap}\n")


def plot_precision_recall_curve(labels, scores, k_value: int, path: str = "", dataset_name="train"):
    copied_labels = labels.clone()
    copied_labels[torch.nonzero(copied_labels > 0)] = 1
    # plot precision recall curve
    print(f"Calculating precision recall curve for k={k_value}...")
    precision, recall, _ = metrics.precision_recall_curve(copied_labels.tolist(), scores.tolist())
    ap = average_precision_score(copied_labels, scores)
    print("AP val is " + str(ap))
    display = PrecisionRecallDisplay.from_predictions(copied_labels.tolist(),
                                                      scores.tolist(),
                                                      name=f"ood vs id",
                                                      color="darkorange"
                                                      )
    _ = display.ax_.set_title("k=" + str(k_value))
    if path:
        plt.savefig(path + f"/anomaly_detection_result/{dataset_name}_dataset_k" + str(k_value) + "_precision_recall.png")
    plt.show()
    return ap


def analyze_roc_curve(labels, scores, desired_tpr=0.95):
    copied_labels = labels.clone()
    copied_labels[torch.nonzero(copied_labels > 0)] = 1
    # plot roc curve
    fpr, tpr, thresholds = metrics.roc_curve(copied_labels.tolist(), scores.tolist())
    threshold_idx = torch.nonzero(torch.tensor(tpr) >= desired_tpr)[0].item()
    threshold = thresholds[threshold_idx]
    chosen_fpr = fpr[threshold_idx]
    chosen_tpr = tpr[threshold_idx]
    return threshold, chosen_fpr, chosen_tpr


def plot_roc_curve(labels, scores, k_value, path: str = "", dataset_name="train"):
    copied_labels = labels.clone()
    copied_labels[torch.nonzero(copied_labels > 0)] = 1
    # plot roc curve
    print(f"Calculating AuC for k={k_value}...")
    fpr, tpr, thresholds = metrics.roc_curve(copied_labels.tolist(), scores.tolist())
    auc = metrics.auc(fpr, tpr)
    print("auc val is " + str(auc))

    RocCurveDisplay.from_predictions(
        copied_labels.tolist(),
        scores.tolist(),
        name=f"ood vs id",
        color="darkorange",
    )

    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("k = " + str(k_value))
    plt.tight_layout()
    if path:
        plt.savefig(path + f"/anomaly_detection_result/{dataset_name}_dataset_k" + str(k_value) + "_ROC.png")
    plt.show()
    return auc


def plot_scores_histograms(scores, labels, k_value, path, dataset_name="train"):
    fg_scores = [scores[ind] for (ind, label) in enumerate(labels) if label > 0]
    bg_scores = [scores[ind] for (ind, label) in enumerate(labels) if label == 0]
    histogram1 = go.Histogram(x=bg_scores, name='normal_scores', marker=dict(color='blue'))
    histogram2 = go.Histogram(x=fg_scores, name='abnormal_scores', marker=dict(color='red'))
    fig = go.Figure(data=[histogram1, histogram2])
    fig.update_layout(
        title=f'Histogram of fg and bg scores k={k_value}',
        xaxis_title='scores',
        yaxis_title='Count'
    )
    fig.write_html(path + f"/anomaly_detection_result/{dataset_name}_dataset_k{k_value}_normality_scores_hist.html")
    fig.show()
