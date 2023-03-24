import os
import matplotlib.pyplot as plt
import numpy as np

def create_boxplot(evaluation, propagated_masks_dir):
    """
    Plot the boxplots of the evaluation metrics for the different methods, and save it in the `results/` folder.
    args:
        propagated_masks_dir: the path to the folder containing the propagated masks
        evaluation: the dictionary containing the evaluation metrics
    """
    patients = os.listdir(os.path.join(propagated_masks_dir, "gt"))
    ticks = ['LV Cavity', 'LV Myocardium']
    methods = ["nnunet", "forward", "backward", "bidirectional"]

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))

    plt.rcParams['font.size'] = 15

    # evaluation["bidirectional"][patient][label][metric]
    avg_dice = {method: {int(label): [np.mean(evaluation[method][patient][label]["dice"]) for patient in patients] for label in [str(x) for x in range(1,3)]} for method in methods}
    avg_msd = {method: {int(label): [np.mean(evaluation[method][patient][label]["msd"]) for patient in patients] for label in [str(x) for x in range(1,3)]} for method in methods}
    avg_hd = {method: {int(label): [np.mean(evaluation[method][patient][label]["hd"]) for patient in patients] for label in [str(x) for x in range(1,3)]} for method in methods}

    #### SUBPLOT 1: DICE SCORE ####
    method1 = ax[0].boxplot([avg_dice["nnunet"][label] for label in range(1,3)], positions=np.array(np.arange(len([avg_dice["nnunet"][label] for label in range(1,3)])))*1.75-0.45, widths=0.25, patch_artist=True)
    method4 = ax[0].boxplot([avg_dice["bidirectional"][label] for label in range(1,3)], positions=np.array(np.arange(len([avg_dice["bidirectional"][label] for label in range(1,3)])))*1.75-0.15, widths=0.25, patch_artist=True)
    method2 = ax[0].boxplot([avg_dice["forward"][label] for label in range(1,3)], positions=np.array(np.arange(len([avg_dice["forward"][label] for label in range(1,3)])))*1.75+0.15, widths=0.25, patch_artist=True)
    method3 = ax[0].boxplot([avg_dice["backward"][label] for label in range(1,3)], positions=np.array(np.arange(len([avg_dice["backward"][label] for label in range(1,3)])))*1.75+0.45, widths=0.25, patch_artist=True)
    ax[0].set_title("Average Dice Score")
    ax[0].set_ylabel("Dice Score")
    define_box_properties(method1, "#2596be", '2D Sparse')
    define_box_properties(method4, "#49be25", 'Bidirectional')
    define_box_properties(method2, "#9925be", 'Forward')
    define_box_properties(method3, "#be4d25", 'Backward')


    #### SUBPLOT 2: MEAN SURFACE DISTANCE ####
    method1 = ax[1].boxplot([avg_msd["nnunet"][label] for label in range(1,3)], positions=np.array(np.arange(len([avg_dice["nnunet"][label] for label in range(1,3)])))*1.75-0.45, widths=0.25, patch_artist=True)
    method4 = ax[1].boxplot([avg_msd["bidirectional"][label] for label in range(1,3)], positions=np.array(np.arange(len([avg_dice["bidirectional"][label] for label in range(1,3)])))*1.75-0.15, widths=0.25, patch_artist=True)
    method2 = ax[1].boxplot([avg_msd["forward"][label] for label in range(1,3)], positions=np.array(np.arange(len([avg_dice["forward"][label] for label in range(1,3)])))*1.75+0.15, widths=0.25, patch_artist=True)
    method3 = ax[1].boxplot([avg_msd["backward"][label] for label in range(1,3)], positions=np.array(np.arange(len([avg_dice["backward"][label] for label in range(1,3)])))*1.75+0.45, widths=0.25, patch_artist=True)
    ax[1].set_title("Average Mean Surface Distance")
    ax[1].set_ylabel("Mean Surface Distance")
    define_box_properties(method1, "#2596be", '2D Sparse')
    define_box_properties(method4, "#49be25", 'Bidirectional')
    define_box_properties(method2, "#9925be", 'Forward')
    define_box_properties(method3, "#be4d25", 'Backward')

    #### SUBPLOT 3: HAUSDORFF DISTANCE ####
    method1 = ax[2].boxplot([avg_hd["nnunet"][label] for label in range(1,3)], positions=np.array(np.arange(len([avg_dice["nnunet"][label] for label in range(1,3)])))*1.75-0.45, widths=0.25, patch_artist=True)
    method4 = ax[2].boxplot([avg_hd["bidirectional"][label] for label in range(1,3)], positions=np.array(np.arange(len([avg_dice["bidirectional"][label] for label in range(1,3)])))*1.75-0.15, widths=0.25, patch_artist=True)
    method2 = ax[2].boxplot([avg_hd["forward"][label] for label in range(1,3)], positions=np.array(np.arange(len([avg_dice["forward"][label] for label in range(1,3)])))*1.75+0.15, widths=0.25, patch_artist=True)
    method3 = ax[2].boxplot([avg_hd["backward"][label] for label in range(1,3)], positions=np.array(np.arange(len([avg_dice["backward"][label] for label in range(1,3)])))*1.75+0.45, widths=0.25, patch_artist=True)
    ax[2].set_title("Average Hausdorff Distance")
    ax[2].set_ylabel("Hausdorff Distance")
    define_box_properties(method1, "#2596be", '2D Sparse')
    define_box_properties(method4, "#49be25", 'Bidirectional')
    define_box_properties(method2, "#9925be", 'Forward')
    define_box_properties(method3, "#be4d25", 'Backward')

    for a in ax:
        # set the x label values
        a.set_xticks(np.arange(0, len(ticks) * 1.75, 1.75), ticks)
        # set the limit for x axis
        a.set_xlim(-1, len(ticks)*1.34)
        a.grid(alpha=0.5, linestyle='--')

    lines = []
    labels = []
    for aa in fig.axes:
        Line, Label = aa.get_legend_handles_labels()
        lines.extend(Line)
        labels.extend(Label)

    # legend in the top left corner
    outpath = "../results/propagation-quality.pdf"
    fig.legend(lines[:4], labels[:4], bbox_to_anchor=(0.24, 0.415))
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    print(f"Figure generated and exported in {outpath}")

    
def define_box_properties(plot, color_code, label):
    """
    Utility function to chenge the attributes and the legend of a boxplot. It sets the outer color of the boxes 
    and the median line to black, while the inner color and the outliers are set to `color_code`.
    args:
        plot: the plot object returned by the `plt.boxplot` function
        color_code: the color code of the inner box
        label: the label of the boxplot
    """	

    for patch, median, outliers in zip(plot['boxes'], plot['medians'], plot['fliers']):
        patch.set_facecolor(color_code)
        patch.set_edgecolor('black')
        median.set_color('black')
        outliers.set(markerfacecolor=color_code , markeredgecolor="None", markersize=5)
        
    # use plot function to draw a small line to name the legend.
    plt.plot([], 's', c=color_code, label=label)