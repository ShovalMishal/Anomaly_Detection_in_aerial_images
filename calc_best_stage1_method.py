
import os
import json
from collections import defaultdict

import matplotlib.pyplot as plt


def find_max_mean_ap_auc(repo_path):
    max_mean_ap = -float('inf')
    max_mean_auc = -float('inf')
    max_mean_ap_file = ""
    max_mean_auc_file = ""

    # Walk through the directory
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if 'mean_ap' in data and 'mean_auc' in data:
                            mean_ap = data['mean_ap']
                            mean_auc = data['mean_auc']

                            if mean_ap > max_mean_ap:
                                max_mean_ap = mean_ap
                                max_mean_ap_file = file_path

                            if mean_auc > max_mean_auc:
                                max_mean_auc = mean_auc
                                max_mean_auc_file = file_path

                except (json.JSONDecodeError, KeyError, ValueError):
                    # Handle potential errors in reading or parsing JSON files
                    pass

    return max_mean_ap_file, max_mean_auc_file, max_mean_ap, max_mean_auc

def extract_results_plot(repo_path):
    results_dict = defaultdict(dict)
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                model_name = os.path.dirname(root).split("/")[-1]
                method_type = os.path.basename(root)
                results_dict[model_name].setdefault(method_type, defaultdict(dict))
                file_name = os.path.basename(file_path)
                stage = file_name[file_name.find("_")+1:file_name.find(".")]
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if 'mean_ap' in data and 'mean_auc' in data:
                        mean_ap = data['mean_ap']
                        mean_auc = data['mean_auc']
                results_dict[model_name][method_type][stage] = {}
                results_dict[model_name][method_type][stage]["mean_ap"] = mean_ap
                results_dict[model_name][method_type][stage]["mean_auc"] = mean_auc

    with open(os.path.join(repo_path, 'model_method_and_stage_to_auc_and_ap.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)
    plt.figure()
    for model_name, model_methods in results_dict.items():
        for method_type, method_data in model_methods.items():
            for stage, stage_data in method_data.items():
                mean_ap = stage_data["mean_ap"]
                mean_auc = stage_data["mean_auc"]
                plt.scatter(mean_ap, mean_auc, label=f"{model_name}_{method_type}_{stage}")
    plt.xlabel("mean ap")
    plt.ylabel("mean auc")
    # plt.legend()
    plt.grid(True)
    plt.title('Stage 1 - Saliency map cooperation results')
    plt.show()







repo_path = '/home/shoval/Documents/Repositories/single-image-bg-detector/results/normalized_gsd/train/'
max_mean_ap_file, max_mean_auc_file, max_mean_ap, max_mean_auc = find_max_mean_ap_auc(repo_path)
print("File with max mean_ap:", max_mean_ap_file)
print("File with max mean_auc:", max_mean_auc_file)
print("Max mean_ap:", max_mean_ap)
print("Max mean_auc:", max_mean_auc)
extract_results_plot(repo_path)
