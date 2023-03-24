import os
import json
import numpy as np
import SimpleITK as sitk
import tqdm

def compute_area_label(mask, label):
    return np.sum(mask == label)

def get_smoothness_dictionary(preds_seq_path, output_dir):
    """
    Generate a dictionary containing the temporal smoothness of a certain sequence and return it. If the dictionary file already exists, 
    it is simply loaded. Otherwise, it is initialized and saved in the provided `output_dir`.
    args:
        preds_seq_path: Path to the folder containing the predictions (sequences! run `process_outputs.py` first)
        output_dir: Path to the folder where the dictionary will be saved
    returns:
        temporal_consistency: Dictionary containing the temporal smoothness of the sequences
    """

    output_json = os.path.join(output_dir, "temporal_consistency.json")

    if not os.path.exists(output_json):
        nnunet_2d_sparse = {patient: {view: {label: {"area": [], "smoothness": []} for label in ['BG', 'LV', "MYO", "LA"]} for view in ['2CH', '4CH']} for patient in range(1,50+1)}
        nnunet_2d_dense = {patient: {view: {label: {"area": [], "smoothness": []} for label in ['BG', 'LV', "MYO", "LA"]} for view in ['2CH', '4CH']} for patient in range(1,50+1)}
        nnunet_3d = {patient: {view: {label: {"area": [], "smoothness": []} for label in ['BG', 'LV', "MYO", "LA"]} for view in ['2CH', '4CH']} for patient in range(1,50+1)}

        temporal_consistency = {"2D_CAMUS_SPARSE": nnunet_2d_sparse, "2D_CAMUS_DENSE": nnunet_2d_dense, "3D_CAMUS": nnunet_3d}

        for i, folder in enumerate(["2D_CAMUS_SPARSE", "2D_CAMUS_DENSE", "3D_CAMUS"]):
            # Iterate over all patients
            for patient_id in tqdm(range(1, 50+1), total=50):
                for view in ['2CH', '4CH']:
                    # Retrieve patient and load sequence
                    patient_name = "patient" + str(patient_id).zfill(4)
                    path2file = os.path.join(preds_seq_path, folder, f"{patient_name}_{view}_sequence.mhd")
                    seq = sitk.GetArrayFromImage(sitk.ReadImage(path2file))

                    area = np.zeros((4, seq.shape[0]))
                    for label_idx, label in enumerate(['BG', 'LV', "MYO", "LA"]):
                        for frame in np.arange(seq.shape[0]):
                            # Compute area
                            area_frame = compute_area_label(seq[frame], label_idx)
                            area[label_idx, frame] = area_frame
                        
                        # Append area to output dictionary
                        temporal_consistency[folder][patient_id][view][label]["area"] = area[label_idx].tolist()

                        # Use area for smoothness
                        d2 = np.diff(area[label_idx], 2)
                        d2_squared = np.square(d2)
                        smoothness = np.trapz(d2_squared)
                        temporal_consistency[folder][patient_id][view][label]["smoothness"] = smoothness

        # Save temporal consistency to file
        with open("temporal_consistency.json", "w") as f:
            json.dump(temporal_consistency, f)

    else:
        with open(output_json, "r") as f:
            temporal_consistency = json.load(f)

    return temporal_consistency