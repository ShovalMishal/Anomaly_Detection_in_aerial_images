import json

import matplotlib.pyplot as plt
import numpy as np


def plot_bar(ranks_dict, category):
    # Width of a bar
    width = 0.35
    # Increase font size
    plt.rcParams.update({'font.size': 14})
    sorted_tt1 = sorted(ranks_dict.items(), key=lambda x: x[1])
    experiments, tt1_ranks = zip(*sorted_tt1)

    # Positions of the bars on the x-axis
    x = np.arange(len(experiments))

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create bar plots
    bars1 = ax.bar(x - width / 2, tt1_ranks, width)

    # Add text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel(r'TT- $1^{st}$ [#samples]')
    # ax.set_xlabel('Novel Class')
    ax.set_title(r'TT- $1^{{st}}$ for all experiments, for category {}'.format(category))
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    ax.set_yscale('log')
    # ax.legend()

    # Function to add labels on the bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Add labels
    add_labels(bars1)

    # Add grid
    ax.yaxis.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Show plot
    plt.tight_layout()
    plt.savefig(f'./tt1s/tt1_all_experiments_category_{category}.pdf')


OOD_ranks_for_All_experiments = {"experiment_1_sub": "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_1_sub/odin/EER_OOD_ranks_dict.json",
                                 "experiment_1": "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_1/odin/EER_OOD_ranks_dict.json",
                                 "experiment_1_vit": "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_1_vit/odin/EER_OOD_ranks_dict.json",
                                 "experiment_1_vit_sub": "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_1_vit_sub/odin/EER_OOD_ranks_dict.json",
                                 "experiment_2_sub": "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_2_sub/odin/EER_OOD_ranks_dict.json",
                                 "experiment_2": "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_2_without_injection/odin/EER_OOD_ranks_dict.json",
                                 "experiment_2_vit": "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_2_vit/odin/EER_OOD_ranks_dict.json",
                                 "experiment_3": "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_3/odin/EER_OOD_ranks_dict.json",
                                 "experiment_3_vit": "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_3_vit/odin/EER_OOD_ranks_dict.json",
                                 "experiment_4": "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_4/odin/EER_OOD_ranks_dict.json",
                                 "experiment_4_vit": "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_4_vit/odin/EER_OOD_ranks_dict.json",
                                 "experiment_5": "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_5/odin/EER_OOD_ranks_dict.json",
                                 "experiment_5_vit": "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_5_vit/odin/EER_OOD_ranks_dict.json",
                                 "experiment_6": "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_6/odin/EER_OOD_ranks_dict.json",
                                 "experiment_6_vit": "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_6_vit/odin/EER_OOD_ranks_dict.json",
                                 "experiment_7": "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_7/odin/EER_OOD_ranks_dict.json",
                                 "experiment_7_vit": "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_7_vit/odin/EER_OOD_ranks_dict.json",
                                 "LOST_gsd_02": "/home/shoval/Documents/Repositories/LOST/ranks_dict_DOTA02.json",
                                 "LOST_gsd_04": "/home/shoval/Documents/Repositories/LOST/ranks_dict_DOTA04.json",
                                 "LOST_gsd_055": "/home/shoval/Documents/Repositories/LOST/ranks_dict_DOTA055.json",
                                 "LOST_gsd_08": "/home/shoval/Documents/Repositories/LOST/ranks_dict_DOTA08.json",
                                 "LOST_gsd_10": "/home/shoval/Documents/Repositories/LOST/ranks_dict_DOTA10.json",
                                 "CUTLER_gsd_02": "/home/shoval/Documents/Repositories/CutLER/cutler/output/ranks_dict_dota_gsd_02.json",
                                 "CUTLER_gsd_04": "/home/shoval/Documents/Repositories/CutLER/cutler/output/ranks_dict_dota_gsd_04.json",
                                 "CUTLER_gsd_055": "/home/shoval/Documents/Repositories/CutLER/cutler/output/ranks_dict_dota_gsd_055.json",
                                 "CUTLER_gsd_08": "/home/shoval/Documents/Repositories/CutLER/cutler/output/ranks_dict_dota_gsd_08.json",
                                 "CUTLER_gsd_10": "/home/shoval/Documents/Repositories/CutLER/cutler/output/ranks_dict_dota_gsd_10.json",
                                 }

experiments_for_each_category = {"small-vehicle" : ["experiment_2", "LOST_gsd_02", "CUTLER_gsd_02"],
                                 "large-vehicle" : ["experiment_1", "experiment_2", "LOST_gsd_02", "CUTLER_gsd_02"],
                                 "ship" : ["experiment_1", "experiment_6", "LOST_gsd_04", "LOST_gsd_055",
                                           "CUTLER_gsd_055", "CUTLER_gsd_10", "CUTLER_gsd_08", "CUTLER_gsd_02", "CUTLER_gsd_04"],
                                 "swimming-pool" : ["experiment_3", "experiment_7", "LOST_gsd_04", "CUTLER_gsd_08",
                                                    "CUTLER_gsd_055", "CUTLER_gsd_10"],
                                 "helicopter" : ["experiment_4", "LOST_gsd_02", "LOST_gsd_055", "CUTLER_gsd_02", "CUTLER_gsd_04",
                                                 "CUTLER_gsd_055"],
                                 "storage-tank" : ["experiment_6", "LOST_gsd_04", "LOST_gsd_055", "CUTLER_gsd_02", "CUTLER_gsd_04", "CUTLER_gsd_055"],
                                 "harbor" : ["experiment_4", "LOST_gsd_04", "LOST_gsd_055", "CUTLER_gsd_055", "CUTLER_gsd_04"],
                                 "plane" : ["experiment_3", "LOST_gsd_04", "LOST_gsd_055", "CUTLER_gsd_055", "CUTLER_gsd_04", "CUTLER_gsd_02"],
                                 "tennis-court": ["experiment_7", "experiment_6", "LOST_gsd_02", "CUTLER_gsd_02"],
                                 "basketball-court" : ["experiment_5", "LOST_gsd_02", "LOST_gsd_04", "CUTLER_gsd_02", "CUTLER_gsd_055"]
}

all_tt1 = {}
for experiment, file_path in OOD_ranks_for_All_experiments.items():
    file = open(file_path)
    data = json.load(file)
    curr_tt1 = {}
    for key, value in data.items():
        curr_tt1[key] = int(value[0])
    all_tt1[experiment] = curr_tt1

categories = set()
for experiment, tt1 in all_tt1.items():
    categories.update(tt1.keys())


for category in experiments_for_each_category.keys():
    rank_per_category = {}
    relevant_experiments = experiments_for_each_category[category]
    for experiment, tt1 in all_tt1.items():
        if category in tt1 and experiment in relevant_experiments:
        # if category in tt1:
            rank_per_category[experiment] = tt1[category]
    plot_bar(rank_per_category, category)

