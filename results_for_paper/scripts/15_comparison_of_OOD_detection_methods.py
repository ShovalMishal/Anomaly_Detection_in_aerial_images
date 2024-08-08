import os
import re

from matplotlib import pyplot as plt


def plot_auc_for_ood_detection_methods():
    experiment_results_path = "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_1/"

    auc_data = {}
    for method_folder in os.listdir(experiment_results_path):
        method_path = os.path.join(experiment_results_path, method_folder)
        auc_file_path = os.path.join(method_path, 'AUC_values.txt')

        if os.path.isfile(auc_file_path):
            with open(auc_file_path, 'r') as file:
                for line in file:
                    match = re.match(r'(.*) stage AUC: (0\.\d+)', line)
                    if match:
                        auc = float(match.group(2))
                        auc_data[method_folder] = auc

    sorted_auc_data = dict(sorted(auc_data.items(), key=lambda item: item[1]))

    methods = list(sorted_auc_data.keys())
    auc_values = list(sorted_auc_data.values())

    # Plotting the results
    plt.scatter(methods, auc_values)
    plt.xticks(range(len(methods)), methods, rotation=90)
    plt.xlabel('OOD Detection Method')
    plt.ylabel('AUC Value')
    plt.title('AUC Values for Different OOD detection Methods')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../figures_for_paper/AUC_values_for_OOD_detection_methods.pdf")
if __name__ == '__main__':
    plot_auc_for_ood_detection_methods()