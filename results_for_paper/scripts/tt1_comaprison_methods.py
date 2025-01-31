import json

import matplotlib.pyplot as plt
import numpy as np


def load_ranks_dict(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    ranks_dict = {}
    for key, value in data.items():
        ranks_dict[key] = int(value[0])+1
    return ranks_dict

file_1 = "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_1_multiscale_fix_aug/odin/OOD_ranks_dict.json"
file_2 = "/home/shoval/Documents/Repositories/LOST/ranks_dict_DOTA_SS_100.json"
file_3 = "/home/shoval/Documents/Repositories/CutLER/cutler/output/ranks_dict_dotav2_ss_100.json"
tt1_ex1 = load_ranks_dict(file_1)
tt1_ex2 = load_ranks_dict(file_2)
tt1_ex3 = load_ranks_dict(file_3)

sorted_tt1_vit = sorted(tt1_ex1.items(), key=lambda x: x[1])
categories, tt1_ex1 = zip(*sorted_tt1_vit)
tt1_ex2 = [int(tt1_ex2[key]) for key in categories]
tt1_ex3 = [int(tt1_ex3[key]) for key in categories]

# Positions of the bars on the x-axis
x = np.arange(len(categories))

# Width of a bar
width = 0.2

# Increase font size
plt.rcParams.update({'font.size': 10})

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))

# Create bar plots
bars1 = ax.bar(x - width , tt1_ex1, width, label='Our Method', color='green')
bars2 = ax.bar(x , tt1_ex2, width, label='LOST', color='blue')
bars3 = ax.bar(x + width, tt1_ex3, width, label='CutLER', color='red')


# Add text for labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel(r'TT- $1^{st}$ [#samples]')
#ax.set_xlabel('Novel Class')
ax.set_title(r'TT- $1^{st}$ for Ex1 vs LOST and CutLER')
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


# Add grid
ax.yaxis.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show plot
plt.tight_layout()
plt.savefig('../tt1s/tt1_experiment1_all_methods_comparison_resnet18.pdf')
