import json

import matplotlib.pyplot as plt
import numpy as np




ex1_vit_file = open(
    '/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_1_vit/odin/EER_OOD_ranks_dict.json')
data = json.load(ex1_vit_file)
tt1_vit = {}
for key, value in data.items():
    tt1_vit[key] = int(value[0])

sorted_tt1_vit = sorted(tt1_vit.items(), key=lambda x: x[1])
categories, tt1_vit = zip(*sorted_tt1_vit)


ex1_resnet_file = open(
    '/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_1_resnet18/odin/EER_OOD_ranks_dict.json')
data = json.load(ex1_resnet_file)
tt1_resnet18 = {}
for key, value in data.items():
    tt1_resnet18[key] = int(value[0])

tt1_resnet18 = [int(tt1_resnet18[key]) for key in categories]

# Positions of the bars on the x-axis
x = np.arange(len(categories))

# Width of a bar
width = 0.35

# Increase font size
plt.rcParams.update({'font.size': 14})

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))

# Create bar plots
bars1 = ax.bar(x - width/2, tt1_vit, width, label='ViT Classifier based OOD detector', color='green')
bars2 = ax.bar(x + width/2, tt1_resnet18, width, label='ResNet18 Classifier based OOD detector', color='blue')

# Add text for labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel(r'TT- $1^{st}$ [#samples]')
#ax.set_xlabel('Novel Class')
ax.set_title(r'TT- $1^{st}$ for ViT and ResNet18 based OOD detectors')
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

# Add grid
ax.yaxis.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show plot
plt.tight_layout()
plt.savefig('../figures_for_paper/EER_point_tt1_experiment1_comparison_vit_resnet.pdf')
