import json

import matplotlib.pyplot as plt
import numpy as np


file = open('/home/shoval/Documents/Repositories/CutLER/cutler/output/ranks_dict_dota_gsd_10.json')
data = json.load(file)
tt1 = []
for key, value in data.items():
    tt1.append(int(value[0]))

categories = list(data.keys())
# sort
categories = [categories[i] for i in np.argsort(tt1)]
tt1 = np.sort(tt1).tolist()


# Positions of the bars on the x-axis
x = np.arange(len(categories))

# Width of a bar
width = 0.35

# Increase font size
plt.rcParams.update({'font.size': 14})

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))

# Create bar plots
bars1 = ax.bar(x - width/2, tt1, width, color='green')
# bars2 = ax.bar(x + width/2, ad, width, label='Anomaly Detection', color='blue')

# Add text for labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel(r'TT- $1^{st}$ [#samples]')
#ax.set_xlabel('Novel Class')
ax.set_title(r'TT- $1^{st}$ for cutler')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.set_yscale('log')
# ax.legend()

# Function to add labels on the bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(int(height)),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Add labels
add_labels(bars1)
# add_labels(bars2)

# Add grid
ax.yaxis.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show plot
plt.tight_layout()
plt.savefig('/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results_for_paper/tt1s/tt1_cutler_gsd_10.pdf')
