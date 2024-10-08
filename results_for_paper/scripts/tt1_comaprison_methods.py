import json

import matplotlib.pyplot as plt
import numpy as np


def load_ranks_dict(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    ranks_dict = {}
    for key, value in data.items():
        ranks_dict[key] = int(value[0])
    return ranks_dict

file_1 = "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_1_multiscale_resnet50_threshold/odin/OOD_ranks_dict.json"
file_2 = "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_1_multiscale_resnet50_threshold/odin/anomaly_ranks_ranks_dict.json"
file_3 = "/home/shoval/Documents/Repositories/LOST/ranks_dict_DOTA_SS.json"
file_4 = "/home/shoval/Documents/Repositories/CutLER/cutler/output/ranks_dict_dotav2_ss.json"
tt1_ex1 = load_ranks_dict(file_1)
tt1_ex2 = load_ranks_dict(file_2)
tt1_ex3 = load_ranks_dict(file_3)
tt1_ex4 = load_ranks_dict(file_4)

sorted_tt1_vit = sorted(tt1_ex1.items(), key=lambda x: x[1])
categories, tt1_ex1 = zip(*sorted_tt1_vit)
tt1_ex2 = [int(tt1_ex2[key]) for key in categories]
tt1_ex3 = [int(tt1_ex3[key]) for key in categories]
tt1_ex4 = [int(tt1_ex4[key]) for key in categories]

# Positions of the bars on the x-axis
x = np.arange(len(categories))

# Width of a bar
width = 0.2

# Increase font size
plt.rcParams.update({'font.size': 10})

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))

# Create bar plots
bars1 = ax.bar(x - 3 * width / 2, tt1_ex1, width, label='Resnet50', color='green')
bars2 = ax.bar(x - width / 2, tt1_ex2, width, label='Anomaly Score', color='blue')
bars3 = ax.bar(x + width / 2, tt1_ex3, width, label='LOST', color='red')
bars4 = ax.bar(x + 3 * width / 2, tt1_ex4, width, label='CutLer', color='orange')


# Add text for labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel(r'TT- $1^{st}$ [#samples]')
#ax.set_xlabel('Novel Class')
ax.set_title(r'TT- $1^{st}$ for Ex1 vs LOST, Cutler, and anomaly score')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.set_yscale('log')
ax.legend()

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
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

# Add grid
ax.yaxis.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show plot
plt.tight_layout()
plt.savefig('../tt1s/EER_point_tt1_experiment1_competitors_comparison.pdf')
