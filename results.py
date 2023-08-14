from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, average_precision_score
import plotly.graph_objects as go


def plot_precision_recall_curve(labels, scores, k_value: int, path: str = ""):
    # plot precision recall curve
    print(f"Calculating precision recall curve for k={k_value}...")
    precision, recall, _ = metrics.precision_recall_curve(labels.tolist(), scores.tolist())
    ap = average_precision_score(labels, scores)
    print("AP val is " + str(ap))
    display = PrecisionRecallDisplay.from_predictions(labels.tolist(),
                                                      scores.tolist(),
                                                      name=f"ood vs id",
                                                      color="darkorange"
                                                      )
    _ = display.ax_.set_title("k=" + str(k_value))
    if path:
        plt.savefig(path + "/statistics/pyramid_func_result/k" + str(k_value) + "_precision_recall.png")
    plt.show()
    return ap


def plot_roc_curve(labels, scores, k_value, path: str = ""):
    # plot roc curve
    print(f"Calculating AuC for k={k_value}...")
    fpr, tpr, thresholds = metrics.roc_curve(labels.tolist(), scores.tolist())
    auc = metrics.auc(fpr, tpr)
    print("auc val is " + str(auc))

    RocCurveDisplay.from_predictions(
        labels.tolist(),
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
        plt.savefig(path + "/statistics/pyramid_func_result/k" + str(k_value) + "_ROC.png")
    plt.show()
    return auc


def plot_scores_histograms(scores, labels, k_value, path):
    abnormal_scores = [scores[ind] for (ind, label) in enumerate(labels) if label==1]
    normal_scores = [scores[ind] for (ind, label) in enumerate(labels) if label < 1]
    histogram1 = go.Histogram(x=normal_scores, name='normal_scores', marker=dict(color='blue'))
    histogram2 = go.Histogram(x=abnormal_scores, name='abnormal_scores', marker=dict(color='red'))
    fig = go.Figure(data=[histogram1, histogram2])
    fig.update_layout(
        title=f'Histogram of fg and bg scores k={k_value}',
        xaxis_title='scores',
        yaxis_title='Count'
    )
    fig.write_html(path + f"/statistics/pyramid_func_result/k{k_value}_normality_scores_hist.html")
    fig.show()
