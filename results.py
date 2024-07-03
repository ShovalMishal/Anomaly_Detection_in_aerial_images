import os
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, average_precision_score, confusion_matrix
import plotly.graph_objects as go


def plot_graphs(scores, labels, path, abnormal_labels, title="", dataset_name="train", ood_mode=False,
                labels_to_classes_names=None, plot_EER=False, bg_scores=None, logger=None):
    os.makedirs(path, exist_ok=True)
    plot_scores_histograms(scores=scores, labels=labels, abnormal_labels=abnormal_labels, title=title, path=path,
                           dataset_name=dataset_name, labels_to_classes_names=labels_to_classes_names,
                           per_label=True)
    ap = plot_precision_recall_curve(labels=labels, scores=scores, abnormal_labels=abnormal_labels, title=title,
                                     path=path, dataset_name=dataset_name, ood_mode=ood_mode)
    auc, eer_threshold = plot_roc_curve(labels=labels, scores=scores, abnormal_labels=abnormal_labels, title=title, path=path,
                         dataset_name=dataset_name, ood_mode=ood_mode, plot_EER=plot_EER, bg_scores=bg_scores,
                         logger=logger, labels_to_classes_names=labels_to_classes_names)
    return auc, ap, eer_threshold


def plot_precision_recall_curve(labels, scores, abnormal_labels, title="", path: str = "", dataset_name="train",
                                ood_mode=False):
    # all abnormal labels are changed to 1
    mask = labels.unsqueeze(1).eq(torch.tensor(abnormal_labels).unsqueeze(0))
    new_labels = torch.any(mask, dim=1).to(torch.int)
    if ood_mode:
        # in ood mode, the lower the score the higher the likelihood the sample is ood! so the abnormal labels
        # in this case, are changed to zero and the normal to 1
        scores = -scores

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
    # plt.show()
    return ap


def analyze_roc_curve(labels, abnormal_labels, scores, desired_tpr=0.95, ood_mode=False):
    # all abnormal labels are changed to 1
    mask = labels.unsqueeze(1).eq(torch.tensor(abnormal_labels).unsqueeze(0))
    new_labels = torch.any(mask, dim=1).to(torch.int)
    if ood_mode:
        # in ood mode, the lower the score the higher the likelihood the sample is ood! so the abnormal labels
        # in this case, are changed to zero and the normal to 1
        scores = -scores
    # plot roc curve
    fpr, tpr, thresholds = metrics.roc_curve(new_labels.tolist(), scores.tolist())
    threshold_idx = torch.nonzero(torch.tensor(tpr) >= desired_tpr)[0].item()
    threshold = thresholds[threshold_idx]
    chosen_fpr = fpr[threshold_idx]
    chosen_tpr = tpr[threshold_idx]
    return threshold, chosen_fpr, chosen_tpr


def plot_roc_curve(labels, scores, abnormal_labels, title="", path: str = "", dataset_name="train", ood_mode=False,
                   plot_EER=False, bg_scores=None, logger=None, labels_to_classes_names=None, rank_accord_features=True):
    # all abnormal labels are changed to 1
    mask = labels.unsqueeze(1).eq(torch.tensor(abnormal_labels).unsqueeze(0))
    new_labels = torch.any(mask, dim=1).to(torch.int)
    if ood_mode:
        # in ood mode, the lower the score the higher the likelihood the sample is ood! so the abnormal labels
        # in this case, are changed to zero and the normal to 1
        scores = -scores
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
    eer_threshold, eer_threshold_idx = calculate_eer_threshold(fpr, tpr, thresholds)
    if plot_EER:
        plot_eer_and_OOD_values_order(fpr, tpr, thresholds, path, title, scores, dataset_name, new_labels, labels,
                                      abnormal_labels, bg_scores, logger, labels_to_classes_names)


    return auc, eer_threshold


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
    plt.figure()
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
    # plt.show()

def calculate_eer_threshold(fpr, tpr, thresholds):
    # Find EER threshold
    eer_threshold_idx = np.argmin(np.abs(fpr - (1 - tpr)))
    eer_threshold = thresholds[eer_threshold_idx]
    return eer_threshold, eer_threshold_idx

def plot_eer_and_OOD_values_order(fpr, tpr, thresholds, path, title, scores, dataset_name, new_labels, labels,
                                  abnormal_labels, bg_scores, logger=None, labels_to_classes_names=None):
    eer_threshold, eer_threshold_idx = calculate_eer_threshold(fpr, tpr, thresholds)
    # Plot EER point
    plt.scatter(fpr[eer_threshold_idx], tpr[eer_threshold_idx], c='red', label=f'EER Threshold ({eer_threshold:.4f})')
    plt.legend()
    if path:
        if "k" in title:
            plt.savefig(path + f"/{dataset_name}_dataset_k" + str(title[-1]) + "_ROC.png")
        else:
            plt.savefig(path + f"/{dataset_name}_dataset_" + str(title) + "_ROC.png")
    # plt.show()

    # Plot Confusion matrix for the EER point
    all_preds = torch.zeros_like(scores)
    all_preds[scores >= eer_threshold] = 1  # OOD
    all_preds[scores < eer_threshold] = 0   # ID
    print_ood_id_outcomes(all_preds, labels, labels_to_classes_names, logger)

    res_confusion_matrix = confusion_matrix(new_labels, all_preds)
    plot_confusion_matrix(confusion_matrix=res_confusion_matrix,
                          classes=["ID", "OOD"],
                          normalize=True,
                          output_dir=path)
    plot_confusion_matrix(confusion_matrix=res_confusion_matrix,
                          classes=["ID", "OOD"],
                          normalize=False,
                          output_dir=path)

    # OOD odin score ranks plot
    high_thresh_odin_scores = scores[scores >= eer_threshold]
    high_thresh_labels = labels[scores >= eer_threshold]

    ood_high_thresh_odin_scores = high_thresh_odin_scores[[label in abnormal_labels for label in high_thresh_labels]]
    sorted_ood_high_thresh_odin_scores, ood_high_thresh_odin_scores_indices = torch.sort(ood_high_thresh_odin_scores)

    sorted_high_thresh_odin_scores, sorted_high_thresh_odin_scores_indices = torch.sort(high_thresh_odin_scores)
    ood_ranks_in_sorted_high_thresh_odin_scores = len(sorted_high_thresh_odin_scores) - 1 - torch.searchsorted(sorted_high_thresh_odin_scores, sorted_ood_high_thresh_odin_scores)

    # find how well OOD samples are ranked in the ODIN scores - per class
    for OOD_label in abnormal_labels:
        if OOD_label not in high_thresh_labels:
            continue
        curr_label_ood_high_thresh_odin_scores = high_thresh_odin_scores[high_thresh_labels == OOD_label]
        curr_label_sorted_ood_high_thresh_odin_scores, curr_label_ood_high_thresh_odin_scores_indices = torch.sort(curr_label_ood_high_thresh_odin_scores)
        curr_ood_label_ranks_in_sorted_high_thresh_odin_scores = len(sorted_high_thresh_odin_scores) - 1 - torch.searchsorted(sorted_high_thresh_odin_scores, curr_label_sorted_ood_high_thresh_odin_scores)
        logger.info(f"OOD label {labels_to_classes_names[OOD_label]} first rank in ODIN scores: {torch.sort(curr_ood_label_ranks_in_sorted_high_thresh_odin_scores)[0][0]}")

    plt.figure()
    plt.plot(list(range(len(ood_ranks_in_sorted_high_thresh_odin_scores))), torch.sort(ood_ranks_in_sorted_high_thresh_odin_scores)[0])
    plt.xlabel('OOD rank in ODIN scores, after stage 2')
    plt.ylabel('post stage 2 rank')
    plt.title('OOD rank')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(path + f"/EER_OOD_rank_in_ODIN_scores.png")
    # plt.show()

    # OOD bg scores ranks plot
    high_thresh_bg_scores = bg_scores[scores >= eer_threshold]
    high_thresh_labels = labels[scores >= eer_threshold]

    ood_high_thresh_bg_scores = high_thresh_bg_scores[[label in abnormal_labels for label in high_thresh_labels]]
    sorted_ood_high_thresh_bg_scores, ood_high_thresh_bg_scores_indices = torch.sort(ood_high_thresh_bg_scores)

    sorted_high_thresh_bg_scores, sorted_high_thresh_bg_scores_indices = torch.sort(high_thresh_bg_scores)
    ood_ranks_in_sorted_high_thresh_bg_scores = torch.searchsorted(sorted_high_thresh_bg_scores, sorted_ood_high_thresh_bg_scores)

    plt.figure()
    plt.plot(list(range(len(ood_ranks_in_sorted_high_thresh_bg_scores))), ood_ranks_in_sorted_high_thresh_bg_scores)
    plt.xlabel('OOD rank in BG scores, after stage 2')
    plt.ylabel('post stage 2 rank')
    plt.title('OOD rank')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(path + f"/EER_OOD_rank_in_BG_scores.png")
    # plt.show()

    # ood odin scores ranks plot
    ood_odin_scores = scores[[label in abnormal_labels for label in labels]]
    sorted_ood_odin_scores, sorted_ood_odin_scores_indices = torch.sort(ood_odin_scores)

    sorted_odin_scores, sorted_odin_scores_indices = torch.sort(scores)
    ood_ranks_in_sorted_ood_scores = len(sorted_odin_scores) - 1 - torch.searchsorted(sorted_odin_scores, sorted_ood_odin_scores)

    plt.figure()
    plt.plot(list(range(len(ood_ranks_in_sorted_ood_scores))), torch.sort(ood_ranks_in_sorted_ood_scores)[0])
    plt.xlabel('OOD ranks in ODIN scores')
    plt.ylabel('all samples ODIN rank')
    plt.title('OOD rank')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(path + f"/OOD_ranks_in_odin_scores.png")
    # plt.show()


def print_ood_id_outcomes(all_preds, labels, labels_to_classes_names, logger):
    # print OOD representation
    OOD_labels = labels[all_preds == 1]
    OOD_labels_unique, OOD_labels_counts = torch.unique(OOD_labels, return_counts=True)
    ood_rep = ("EER OOD representation is:\n")
    for label, count in zip(OOD_labels_unique, OOD_labels_counts):
        ood_rep+=f"Class {labels_to_classes_names[label.item()]} shown {count.item()} times\n"
    logger.info(ood_rep)

    # print ID representation
    ID_labels = labels[all_preds == 0]
    ID_labels_unique, ID_labels_counts = torch.unique(ID_labels, return_counts=True)
    id_rep = ("EER ID representation is:\n")
    for label, count in zip(ID_labels_unique, ID_labels_counts):
        id_rep+=f"Class {labels_to_classes_names[label.item()]} shown {count.item()} times \n"
    logger.info(id_rep)


def show_50_outliers():
    # Path to the folder containing the images
    folder_path = "./50_highest_scores_patches"

    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

    # Define the number of images to display per row and column
    num_images_per_row = 10
    num_images_per_col = 5

    # Create a new figure
    fig, axs = plt.subplots(num_images_per_col, num_images_per_row, figsize=(16, 8))

    # Loop through the images and display them on the plot
    for i, file in enumerate(image_files):
        # Calculate the position of the image in the grid
        row = i // num_images_per_row
        col = i % num_images_per_row

        # Load the image using PIL
        img = Image.open(os.path.join(folder_path, file))

        # Display the image on the plot
        axs[row, col].imshow(img)
        axs[row, col].axis('off')

        # Set the title of the image as its name
        axs[row, col].set_title(file[:file.find("_")], fontsize=8)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    # Show the plot
    plt.show()
