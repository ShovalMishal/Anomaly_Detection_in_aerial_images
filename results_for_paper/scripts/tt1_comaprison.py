import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_data(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def print_ood_summary(data):
    ood_num=0
    for key, value in data.items():
        print(f'{key} is shown {len(value)} times\n')
        ood_num+=len(value)
    print(f'ood objects number is {ood_num}\n')


def load_ranks_dict(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    if "AD" in os.path.basename(file_path):
        print("Anomaly detection summary:\n")
    else:
        print("OOD summary:\n")
    print_ood_summary(data)
    ranks_dict = {}
    for key, value in data.items():
        ranks_dict[key] = int(value[0]) + 1
    return ranks_dict

file_2 = "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/anomaly_detection_result/experiment_2_multiscale_fix_aug/AD_ranks_dict.json"
file_1 = "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/experiment_2_multiscale_fix_aug/odin/OOD_ranks_dict.json"

tt1_ex1 = load_ranks_dict(file_1)
tt1_ex2 = load_ranks_dict(file_2)

sorted_tt1_vit = sorted(tt1_ex1.items(), key=lambda x: x[1])
categories, tt1_ex1 = zip(*sorted_tt1_vit)
tt1_ex2 = [int(tt1_ex2[key]) for key in categories]

# Positions of the bars on the x-axis
x = np.arange(len(categories))

# Width of a bar
width = 0.35

# Increase font size
plt.rcParams.update({'font.size': 14})

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))

# Create bar plots
bars1 = ax.bar(x - width / 2, tt1_ex1, width, label='Out-Of-Distribution', color='green')
bars2 = ax.bar(x + width / 2, tt1_ex2, width, label='Anomaly Detection', color='blue')


# Add text for labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel(r'TT- $1^{st}$ [#samples]')
#ax.set_xlabel('Novel Class')
ax.set_title(r'TT- $1^{st}$ for OOD and AD')
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
plt.savefig('../tt1s/experiment2_AD_vs_OOD.pdf')
