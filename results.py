import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, average_precision_score
import plotly.graph_objects as go


def plot_graphs(scores, labels, path, abnormal_labels, title="", dataset_name="train", ood_mode=False, labels_to_classes_names={}):
    os.makedirs(path, exist_ok=True)
    plot_scores_histograms(scores=scores, labels=labels, abnormal_labels=abnormal_labels, title=title, path=path,
                           dataset_name=dataset_name, labels_to_classes_names=labels_to_classes_names, per_label=True)
    ap = plot_precision_recall_curve(labels=labels, scores=scores, abnormal_labels=abnormal_labels, title=title,
                                     path=path, dataset_name=dataset_name, ood_mode=ood_mode)
    auc = plot_roc_curve(labels=labels, scores=scores, abnormal_labels=abnormal_labels, title=title, path=path,
                         dataset_name=dataset_name, ood_mode=ood_mode)
    return auc, ap


def plot_precision_recall_curve(labels, scores, abnormal_labels, title="", path: str = "", dataset_name="train",
                                ood_mode=False):
    # all abnormal labels are changed to 1
    mask = labels.unsqueeze(1).eq(torch.tensor(abnormal_labels).unsqueeze(0))
    new_labels = torch.any(mask, dim=1).to(torch.int)
    if ood_mode:
        # in ood mode, the lower the score the higher the likelihood the sample is ood! so the abnormal labels
        # in this case, are changed to zero and the normal to 1
        new_labels = new_labels ^ 1

    # plot precision recall curve
    print(f"Calculating precision recall curve {title}...")
    precision, recall, _ = metrics.precision_recall_curve(new_labels.tolist(), scores.tolist())
    ap = average_precision_score(new_labels, scores)
    print("AP val is " + str(ap))
    display = PrecisionRecallDisplay.from_predictions(new_labels.tolist(),
                                                      scores.tolist(),
                                                      name=f"ood vs id",
                                                      color="darkorange"
                                                      )
    _ = display.ax_.set_title(title)
    if path:
        if "k" in title:
            plt.savefig(path + f"/{dataset_name}_dataset_k" + str(title[-1]) + "_precision_recall.png")
        else:
            plt.savefig(path + f"/{dataset_name}_dataset_" + str(title) + "_precision_recall.png")
    plt.show()
    return ap


def analyze_roc_curve(labels, abnormal_labels, scores, desired_tpr=0.95, ood_mode=False):
    # all abnormal labels are changed to 1
    mask = labels.unsqueeze(1).eq(torch.tensor(abnormal_labels).unsqueeze(0))
    new_labels = torch.any(mask, dim=1).to(torch.int)
    if ood_mode:
        # in ood mode, the lower the score the higher the likelihood the sample is ood! so the abnormal labels
        # in this case, are changed to zero and the normal to 1
        new_labels = new_labels ^ 1
    # plot roc curve
    fpr, tpr, thresholds = metrics.roc_curve(new_labels.tolist(), scores.tolist())
    threshold_idx = torch.nonzero(torch.tensor(tpr) >= desired_tpr)[0].item()
    threshold = thresholds[threshold_idx]
    chosen_fpr = fpr[threshold_idx]
    chosen_tpr = tpr[threshold_idx]
    return threshold, chosen_fpr, chosen_tpr


def plot_roc_curve(labels, scores, abnormal_labels, title="", path: str = "", dataset_name="train", ood_mode=False):
    # all abnormal labels are changed to 1
    mask = labels.unsqueeze(1).eq(torch.tensor(abnormal_labels).unsqueeze(0))
    new_labels = torch.any(mask, dim=1).to(torch.int)
    if ood_mode:
        # in ood mode, the lower the score the higher the likelihood the sample is ood! so the abnormal labels
        # in this case, are changed to zero and the normal to 1
        new_labels = new_labels ^ 1
    # plot roc curve
    print(f"Calculating AuC for {title}...")
    fpr, tpr, thresholds = metrics.roc_curve(new_labels.tolist(), scores.tolist())
    auc = metrics.auc(fpr, tpr)
    print("auc val is " + str(auc))

    RocCurveDisplay.from_predictions(
        new_labels.tolist(),
        scores.tolist(),
        name=f"ood vs id",
        color="darkorange",
    )

    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.tight_layout()
    if path:
        if "k" in title:
            plt.savefig(path + f"/{dataset_name}_dataset_k" + str(title[-1]) + "_ROC.png")
        else:
            plt.savefig(path + f"/{dataset_name}_dataset_" + str(title) + "_ROC.png")
    plt.show()
    return auc


def plot_scores_histograms(scores, labels, abnormal_labels, title="", path="", dataset_name="train", per_label=False,
                           labels_to_classes_names={}):
    scores = scores.tolist()
    labels = labels.tolist()
    if not per_label:
        abnormal_scores = [scores[ind] for (ind, label) in enumerate(labels) if label in abnormal_labels]
        normal_scores = [scores[ind] for (ind, label) in enumerate(labels) if label not in abnormal_labels]
        histogram1 = go.Histogram(x=normal_scores, name='normal_scores', marker=dict(color='blue'))
        histogram2 = go.Histogram(x=abnormal_scores, name='abnormal_scores', marker=dict(color='red'))
        fig = go.Figure(data=[histogram1, histogram2])
        fig.update_layout(
            title=f'Histogram of normal and abnormal scores {title}',
            xaxis_title='scores',
            yaxis_title='Count'
        )
    else:
        all_hists = []
        labels_set = set(labels)
        for curr_label in labels_set:
            curr_labels_scores = [scores[ind] for (ind, label) in enumerate(labels) if label==curr_label]
            curr_labels_hist = go.Histogram(x=curr_labels_scores, name=f'{labels_to_classes_names[curr_label]}')
            all_hists.append(curr_labels_hist)
        fig = go.Figure(data=all_hists)
        fig.update_layout(
            title=f'Histogram of all labels scores {title}',
            xaxis_title='scores',
            yaxis_title='Count'
        )
    if path:
        if "k" in title:
            fig.write_html(path + f"/{dataset_name}_dataset_k{str(title[-1])}_normality_scores_hist.html")
        else:
            fig.write_html(path + f"/{dataset_name}_dataset_{str(title)}_normality_scores_hist.html")
    fig.show()


def plot_confusion_matrix(confusion_matrix, classes, normalize=True, output_dir="./"):
    correct = 0
    for i in range(len(confusion_matrix)):
        correct += confusion_matrix[i, i]
    correct / confusion_matrix.sum()
    acc = correct / confusion_matrix.sum() * 100
    normalized_confusion_matrix = confusion_matrix.astype(float)
    if normalize:
        for idx in range(len(normalized_confusion_matrix)):
            normalized_confusion_matrix[idx, ...] = normalized_confusion_matrix[idx, ...] / normalized_confusion_matrix[idx, ...].sum()
    title = " normalized " if normalize else " "
    file_name = "normalized_" if normalize else ""
    plt.imshow(normalized_confusion_matrix)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, f'{confusion_matrix[i, j]}', ha='center', va='center', color='white')
    plt.colorbar()
    plt.xticks(range(len(classes)), classes, rotation=90)
    plt.yticks(range(len(classes)), classes)
    plt.title(
        f'confusion matrix{title}according to rows\naccuracy: {acc:.2f}[%]')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_name}training_confusion_matrix.png"))
    plt.show()

