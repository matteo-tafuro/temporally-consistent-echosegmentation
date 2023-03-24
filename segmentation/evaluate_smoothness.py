import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.utils import retrieve_ED_ES_frames, retrieve_ED_ES_preds, retrieve_gt_area
from utils.plots import split_axis, plot_contour_class
from utils.eval import get_smoothness_dictionary

def plot_temporal_consistency(smoothness_dict, output_dir, models_colors = ["#2596be", "#EAA70B", "#be2596"]):
    """
    Plot the temporal consistency of the outputs of each model (as seen in Fig. 3 of the paper) and save it in `output_dir`.
    args:
        smoothness_dict: Dictionary containing the smoothness values for each model
        output_dir: Path to the output directory
        models_colors: List of colors for each model
    """

    plt.rcParams['font.size'] = 20
    fig, ax = plt.subplots(1, 3, figsize=(20, 6), dpi=200)

    for i, (label, legend) in enumerate(zip(['LV', "MYO", "LA"], ['Left Ventricle Cavity', "Left Ventricle Myocardium", "Left Atrium"])):

        bxplot = ax[i].boxplot([smoothness_dict["2D_CAMUS_SPARSE"][label], smoothness_dict["2D_CAMUS_DENSE"][label], smoothness_dict["3D_CAMUS"][label]], patch_artist=True)
        for patch, median, outliers, color in zip(bxplot['boxes'], bxplot['medians'], bxplot['fliers'], models_colors):
            patch.set_facecolor(color)
            median.set_color('black')
            # median.set_linewidth(3)
            outliers.set(markerfacecolor=color, markeredgecolor=None, markeredgewidth=0)

        for model, color in zip(['2D Sparse', '2D Dense', '3D'], models_colors):
            plt.plot([], 's', c=color, label=model)
        ax[i].set_title(legend)

    lines = []
    labels = []
    for aa in fig.axes:
        Line, Label = aa.get_legend_handles_labels()
        # print(Label)
        lines.extend(Line)
        labels.extend(Label)
    ax[0].set_ylabel(r"Smoothness")
    for a in ax:
        a.set_xticklabels(['2D Sparse', '2D Dense', '3D'])
        a.set_yscale('log')
        a.grid(alpha=0.5, linestyle='--')

    # legend in the top left corner
    fig.legend(lines[:3], labels[:3], loc='upper right', bbox_to_anchor=(0.94, 0.86))

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "smoothness-models.pdf"), bbox_inches='tight')


def plot_smoothness_curves(patient_id, view, output_dir, keep_original_ratio=False):
    """
    Plot the area curves with corresponding predictions at ED and ES for the specified patient and view, then save the figure in the specified directory.
    This script assumes that the predictions of the three models are already saved in the `segmentation/nnUNet/seq_predictions` folder, as
    outputted by the `process_output.py` script. By default, the script assumes that the aforementioned directory contains the following subdirectories:
    - 2D_CAMUS_SPARSE
    - 2D_CAMUS_DENSE
    - 3D_CAMUS
    args:
        patient_id: Patient ID
        view: View ID
        output_dir: Path to the directory where the figure will be saved
        keep_original_ratio: If True, the images are resized to their original size. Otherwise, the images are kept as 512x512.
    """
    
    output_dir = os.path.join(output_dir, "temporal_consistency_curves")
    os.makedirs(output_dir, exist_ok=True)

    patient_name = "patient" + str(patient_id).zfill(4)

    markers = [".-", ".-", ".-"]

    fig, ax = plt.subplots(2, 3, figsize=(20, 10), dpi=200)
    plt.rcParams.update({'font.size': 20})

    gt_area_ED, gt_area_ES = retrieve_gt_area(patient_name, view)
    (ED_frame, ED_mask), (ES_frame, ES_mask), ED_size, ES_size = retrieve_ED_ES_frames(patient_name, view, keep_original_ratio=keep_original_ratio)

    gt_cmap = mpl.colors.ListedColormap(["#25be4d"])

    for label_id, label in enumerate(['LV', "MYO", "LA"]):
        ax_1, ax_2 = None, None
        for i, (model, color) in enumerate(zip(["2D_CAMUS_SPARSE", "2D_CAMUS_DENSE", "3D_CAMUS"], ["#2596be", "#EAA70B", "#be2596"])):

            area = temporal_consistency[model][str(patient_id)][view][label]["area"]
            x_axis_area = np.arange(1, len(area) + 1)
            ax[0,label_id].plot(x_axis_area, area, markers[i], label=label, color=color, linewidth=4, ms=15,)
            ax[0,label_id].plot(x_axis_area[0], gt_area_ED[label_id+1], "X", color="#25be4d", ms=15, zorder=10, clip_on=False, label="GT area at ED")
            ax[0,label_id].plot(x_axis_area[-1], gt_area_ES[label_id+1], "X", color="#25be4d", ms=15, zorder=10, clip_on=False, label="GT area at ES")

            
            # Split axis and plot ED and ES frames
            if ax_1 is None:
                ax_1, ax_2 = split_axis(ax[1,label_id], pad=0.6 if keep_original_ratio else 0.05)
                ax_1.imshow(ED_frame, cmap="gray", vmin=0, vmax=1)
                ED_masked = np.ma.masked_where(ED_mask != label_id+1, ED_mask)
                ax_1.imshow(ED_masked, cmap=gt_cmap, alpha=1)
                ax_2.imshow(ES_frame, cmap="gray", vmin=0, vmax=1)
                ES_masked = np.ma.masked_where(ES_mask != label_id+1, ES_mask)
                ax_2.imshow(ES_masked, cmap=gt_cmap, alpha=1)
            if model == "3D_CAMUS":
                if keep_original_ratio:
                    ED_pred, ES_pred = retrieve_ED_ES_preds(patient_name, view, model, seq_path='nnUNet/seq_predictions', ED_size=ED_size, ES_size=ES_size)
                else:
                    ED_pred, ES_pred = retrieve_ED_ES_preds(patient_name, view, model, seq_path='nnUNet/seq_predictions')
                plot_contour_class(ax_1, ED_pred, label_id+1, color=color, thickness=12 if keep_original_ratio else 10)
                ax_1.set_title("ED")
                plot_contour_class(ax_2, ES_pred, label_id+1, color=color, thickness=12 if keep_original_ratio else 10)
                ax_2.set_title("ES")
            # Remove all axis ticks from ax_1 and ax_2
            ax_1.set_xticks([])
            ax_1.set_yticks([])
            ax_2.set_xticks([])
            ax_2.set_yticks([])
            
    # Fix titles
    ax[0,0].set_title("LV Cavity")
    ax[0,1].set_title("LV Myocardium")
    ax[0,2].set_title("Left Atrium")

    lines=[]
    labels=[]
    for aa in fig.axes:
        Line, Label = aa.get_legend_handles_labels()
        lines.extend(Line)
        labels.extend(Label)

    # legend in the top left corner
    indices = np.array([0,3,6,1,2])
    fig.legend([lines[i] for i in indices], ["2D Sparse", "2D Dense", "3D", "Ground truth"], loc='upper right', bbox_to_anchor=(0.212, 0.745 if keep_original_ratio else 0.72))

    xticks = np.linspace(1, x_axis_area[-1], 6, dtype=int)

    [ax[0, i].set_xticks(xticks) for i in range(3)]
    [ax[0, i].tick_params(axis='x',which='minor',direction='out',bottom=True,length=5) for i in range(3)]
    [ax[0, i].set_xlim([np.min(x_axis_area),np.max(x_axis_area)]) for i in range(3)]
    [ax[0, i].grid(True, alpha=0.5) for i in range(3)]
    [ax[0, i].set_xlabel("Frame") for i in range(3)]
    [ax[0, 0].set_ylabel("Area [px]") for i in range(3)]
    [ax[0, i].ticklabel_format(axis='y', style='sci', scilimits=(0,0)) for i in range(3)]

    plt.tight_layout()

    fig.savefig(os.path.join(output_dir, f"temporal_consistency_curve_{patient_id}_{view}.pdf"), bbox_inches="tight")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_id", type=int, required=True,
                        help="Patient ID")
    parser.add_argument("--view", type=str, required=True,
                        help="View (2CH, 4CH)")
    parser.add_argument("--preds_seq_path", type=str, required=True)
    parser.add_argument("--eval_dict_path", type=str, default="../data/propagated_masks",
                        help="Path to the dictionary containing the evaluation of the segmentation smoothness")
    parser.add_argument("--results_path", type=str, default="../results",
                        help="Path to the directory where the temporal consistency curve will be saved")
    parser.add_argument("--keep_original_ratio", action="store_true", default=False,
                        help="If set, the temporal consistency curve will be plotted with the original aspect ratio")
    args = parser.parse_args()

    # Get the temporal consistency dictionary
    temporal_consistency = get_smoothness_dictionary(args.preds_seq_path, args.eval_dict_path)

    # Make a boxplot with general statistics about the smoothness of each model
    plot_temporal_consistency(temporal_consistency, args.results_path)

    # Plot temporal consistency curve for the chosen patient and view
    plot_smoothness_curves(args.patient_id, args.view, args.results_path, args.keep_original_ratio)