import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
from eval import compute_area_label, compute_geometric_metrics

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