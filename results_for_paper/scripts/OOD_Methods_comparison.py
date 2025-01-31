import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    OOD_methods = ["MSP", "Energy", "ODIN", "ViM"]
    OOD_methods_scores = [0.7465,0.7515,0.7584, 0.7514]
    width = 0.35
    plt.rcParams.update({'font.size': 14})
    x = np.arange(len(OOD_methods))
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create bar plots
    bars1 = ax.bar(x - width / 2, OOD_methods_scores, width, color='green')
    ax.set_ylabel(r'AUC Value')
    ax.set_title(r'AUC Values for OOD Detection Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(OOD_methods, rotation=45, ha='right')


    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    add_labels(bars1)
    ax.yaxis.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('../figures_for_paper/OOD_Methods_Comparison.pdf')

