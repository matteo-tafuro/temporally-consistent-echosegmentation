import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from TorchIR.utils.eval import compute_area_label, compute_geometric_metrics
from tqdm import tqdm

def evaluate_propagated_masks(propagated_masks_dir, propagation_method, TED_path, ted2camus_path, size=(512,512), overwrite=False):
    
    # Process the TED dataset to get the ground truth masks
    print("Preparing the ground truth masks...")
    prepare_ground_truths(TED_path=TED_path,
                          ted2camus_path=ted2camus_path,
                          size=size,
                          overwrite=overwrite)
    
    # Evaluate all the propagated masks and store the results in a dict
    print("Evaluating the propagated masks...")
    evaluation_dict = get_evaluation_dictionary(propagated_masks_dir=propagated_masks_dir, propagation_method=propagation_method)

    # Use the dictionary to generate and save the plots
    print("Generating the plots...")
    create_boxplot(evaluation_dict, propagated_masks_dir)

def prepare_ground_truths(TED_path, ted2camus_path, size=(512,512), overwrite=False):
    """
    Copy the ground truth masks from the TED dataset to the `propagated_masks` folder, and do the following to the sequence:
        - Trim the video. We only need half a cycle
        - Resize to `512 x 512` for easy comparison
        - Rename TED IDs to CAMUS IDs
    args:
        TED_path: path to the TED dataset
        ted2camus_path: path to the ted2camus.csv file
        size: tuple of the desired size (W, H)
        overwrite: if True, overwrite the folder if it already exists
    """

    output_path = os.path.join(os.path.dirname(os.path.dirname(TED_path)), "propagated_masks", "gt")
    os.makedirs(output_path, exist_ok=True)

    if os.path.exists(output_path) and not overwrite:
        print("Ground truth folder already exists, thereby its generation is skipped. Set `overwrite` to True to overwrite it.")
        return

    # Extract required patients
    ted2camus = pd.read_csv(ted2camus_path)
    ted_ids = list(ted2camus["ted_id"])
    camus_ids = list(ted2camus["camus_id"])

    # Copy files in the new folder
    for ted_id, camus_id in tqdm(zip(ted_ids, camus_ids), total=len(ted_ids)):

        # Output folder
        patient_dir = os.path.join(output_path, f"patient{camus_id:04d}")
        os.makedirs(patient_dir, exist_ok=True)

        # Read ground truth file
        img_itk = sitk.ReadImage(os.path.join(TED_path, f"patient{ted_id:03d}", f"patient{ted_id:03d}_4CH_sequence_gt.mhd"))

        # Read info .CFG file to get the ES frame
        with open(os.path.join(TED_path, f"patient{ted_id:03d}", f"patient{ted_id:03d}_4CH_info.cfg"), "r") as f:
            lines = f.readlines()
            es_frame = int(lines[1].split(" ")[1])

        old_spacing = np.array(img_itk.GetSpacing())
        old_dim = np.array(img_itk.GetSize())
        new_dim = np.array([size[0], size[1], old_dim[2]])

        img = torch.from_numpy(sitk.GetArrayFromImage(img_itk)[:es_frame]).unsqueeze(0)
        img = F.interpolate(img, size=size, mode='nearest-exact')
        img = img.squeeze().numpy()

        # Compute new spacing
        spacing = old_spacing * old_dim / new_dim

        img = (img).astype(np.uint8)

        processed_itk = sitk.GetImageFromArray(img)
        processed_itk.SetSpacing(spacing)
        
        outpath = os.path.join(patient_dir, f"patient{camus_id:04d}_4CH_sequence_gt.mhd")
        sitk.WriteImage(processed_itk, outpath)


def get_evaluation_dictionary(propagated_masks_dir, propagation_method='sequential'):
    """
    Generate the evaluation dictionary (defined as seen below) and return it. Each value array will be as long as the number of frames in the sequence.
    If the dictionary file already exists, it is simply loaded. Otherwise, it is initialized and saved in the `propagated_masks/` folder.
    args:
        propagated_masks_dir: path to the folder containing the propagated masks. By deafult, it should be `data/propagated_masks/`.
        propagation_method: either 'sequential' or 'fixed'.
    returns:
        evaluation: the evaluation dictionary.
    dictionary structure:
        `evaluation = {
            forward: {
                patient0001: {
                    0 (class): {
                        dice: [...],
                        msd: [...],
                        rms: [...],
                        hd: [...]
                        area: [...]
                    },
                    1: {
                        dice: [...],
                        msd: [...],
                        rms: [...],
                        hd: [...]
                        area: [...]
                    },
                    ...
                }
                patient0001: {
                    0: {
                        dice: [...],
                        msd: [...],
                        rms: [...],
                        hd: [...]
                        area: [...]
                    },
                    1: {
                        dice: [...],
                        msd: [...],
                        rms: [...],
                        hd: [...]
                        area: [...]
                    },
                    ...
                },
            backward: {
                ...
            },
            ...
        }`
    """

    # Path to the folder containing the three folders (gt, forward, backward)
    methods = ["forward", "backward", "bidirectional", "nnunet"] 
    methods_path = {
        "forward":          os.path.join(propagated_masks_dir, propagation_method, "forward"),
        "backward":         os.path.join(propagated_masks_dir, propagation_method, "backward"),
        "bidirectional":    os.path.join(propagated_masks_dir, propagation_method, "bidirectional"),
        "nnunet":           os.path.join(propagated_masks_dir, "nnunet"),
        "gt":               os.path.join(propagated_masks_dir, "gt")
    }
    
    patients = os.listdir(os.path.join(propagated_masks_dir, "gt"))

    # Output path
    output_json = os.path.join(propagated_masks_dir, "evaluation.json")

    if os.path.exists(output_json):
        print("Existing evaluation dictionary found, loading it...")
        with open(output_json, "r") as f:
            evaluation = json.load(f)
    
    else:
        print("No evaluation dictionary found, initializing it...")
        evaluation = {method: {patient: {label: {"dice": [], "msd": [], "rms": [], "hd": [], "area": []} if label!=3 else {"area": []} for label in range(0, 4)} for patient in patients} for method in methods}

        i = 0
        for patient in tqdm(patients):
            gt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(propagated_masks_dir, "gt", patient, f"{patient}_4CH_sequence_gt.mhd")))
            for method in methods:
                pred = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(methods_path[method], patient, f"{patient}_4CH_sequence_gt_{method}.mhd")))
                for frame in np.arange(pred.shape[0]):
                    for label in range(0, 3):
                        dice, msd, rms, hd, area = compute_geometric_metrics(pred[frame], gt[frame], label)
                        evaluation[method][patient][label]["dice"].append(dice)
                        evaluation[method][patient][label]["msd"].append(msd)
                        evaluation[method][patient][label]["rms"].append(rms)
                        evaluation[method][patient][label]["hd"].append(hd)
                        evaluation[method][patient][label]["area"].append(area)
                    # We don't have GT masks for the atrium, so we only compute the area
                    evaluation[method][patient][3]["area"].append(compute_area_label(pred[frame], 3))


        # Compute the area of the GT masks
        gt_area_dict = {patient: {label: {"area": []} for label in range(0, 4)} for patient in patients}
        for patient in tqdm(patients):
            gt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(methods_path["gt"], patient, f"{patient}_4CH_sequence_gt.mhd")))
            pred = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(methods_path["forward"], patient, f"{patient}_4CH_sequence_gt_forward.mhd")))
            for frame in np.arange(pred.shape[0]):
                for label in range(0, 3):
                    area = compute_area_label(gt[frame], label)
                    gt_area_dict[patient][label]["area"].append(area)
        evaluation["gt"] = gt_area_dict

        # Save the results in a JSON file
        # Little helper class to serialize numpy arrays
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)

        # save evaluation dictionary
        with open(output_json, "w") as f:
            json.dump(evaluation, f, cls=NpEncoder)

    return evaluation


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


    

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--propagated_masks_dir', type=str, default='../data/propagated_masks', help='path to the directory containing the propagated masks')
    parser.add_argument('--propagation_method', type=str, default='sequential', help='propagation method')
    parser.add_argument('--TED_path', type=str, default='../data/ted/database', help='path to the TED database')
    parser.add_argument('--ted2camus_path', type=str, default='../data/ted2camus.csv', help='path to the ted2camus.csv file')
    parser.add_argument('--size', type=tuple, default=(512,512), help='size of the images')
    parser.add_argument('--overwrite', type=bool, default=False, help='overwrite the existing files')
    args = parser.parse_args()

    evaluate_propagated_masks(
            propagated_masks_dir = args.propagated_masks_dir, 
            propagation_method = args.propagation_method, 
            TED_path = args.TED_path,
            ted2camus_path = args.ted2camus_path, 
            size = args.size, 
            overwrite = args.overwrite,
        )