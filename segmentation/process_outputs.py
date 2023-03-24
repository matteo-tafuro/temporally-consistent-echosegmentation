import os
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from utils.utils import convert_individual_preds_to_seq

def convert_for_submission(predictions_path, original_test_path, pred_2d_or_3d):
    """
    Convert the outputs of nnUNet to the format required for submission (MHD files). The converted files
    will be saved in `nnUNet/submissions`.
    args:
        predictions_path: Path to the folder containing the predictions
        original_test_path: Path to the original test set
        pred_2d_or_3d: whether the predictions are '2d' (single frame) or '3d' (full sequences)
    """

    assert pred_2d_or_3d.lower() in ["2d", "3d"]

    # Create folder in `submission` with the same task name
    output_path = "nnUNet/submissions"

    submission_path = os.path.join(output_path, os.path.basename(predictions_path))
    os.makedirs(submission_path, exist_ok=True)

    pred_files = [file for file in os.listdir(predictions_path) if file.endswith(".nii.gz")]


    for prediction in pred_files:
        # Retrieve original test file to resize prediction and set the correct spacing
        patient = prediction.split("_")[0]
        view = prediction.split("_")[1]

        test_filepath = os.path.join(original_test_path, patient, prediction.replace(".nii.gz", ".mhd"))
        original = sitk.ReadImage(test_filepath)

        og_spacing = np.array(original.GetSpacing())
        og_dim = tuple(original.GetSize()[:2])

        # Resize to original shape
        pred_resized = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(predictions_path, prediction)))).unsqueeze(0)
        pred_resized = F.interpolate(pred_resized, size=og_dim[::-1], mode='nearest-exact')

        if pred_2d_or_3d == "2d": # If the prediction is 2D, we resize it to match the original shape and export it as mhd

            pred = pred_resized.numpy()[:,0]

            # Convert to SimpleITK image
            pred = sitk.GetImageFromArray(pred)

            assert pred.GetSize() == original.GetSize()

            # Set spacing
            pred.SetSpacing(og_spacing)

            # Export prediction
            filename = prediction.replace(".nii.gz", ".mhd")
            sitk.WriteImage(pred, os.path.join(submission_path, filename))

        elif pred_2d_or_3d == "3d": # If the prediction is 3D, we only need the ED and ES frames

            pred_resized = pred_resized.numpy()
    
            # Extract the frames we need
            pred_ED = pred_resized[:,0]
            pred_ES = pred_resized[:,-1]

            # Convert to SimpleITK image
            pred_ED = sitk.GetImageFromArray(pred_ED)
            pred_ES = sitk.GetImageFromArray(pred_ES)

            # Set spacing
            pred_ED.SetSpacing(og_spacing)
            pred_ES.SetSpacing(og_spacing)

            # Export prediction
            ED_filename = f'{patient}_{view}_ED.mhd'
            ES_filename = f'{patient}_{view}_ES.mhd'
            sitk.WriteImage(pred_ED, os.path.join(submission_path, ED_filename))
            sitk.WriteImage(pred_ES, os.path.join(submission_path, ES_filename))


if __name__ == "__main__":

    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--predictions_path", type=str, required=True, help="Path to the folder containing the predictions")
    args.add_argument("--original_test_path", type=str, default="../data/camus/testing", help="Path to the original test set")
    args.add_argument("--pred_2d_or_3d", type=str, required=True, help="Whether the predictions are '2d' (single frame) or '3d' (full sequences)")
    args = args.parse_args()

    # Convert .nii.gz to .mhd for submission platform
    convert_for_submission(args.predictions_path, args.original_test_path, args.pred_2d_or_3d)

    # Convert individual predictions to sequences, in order to allow for the evaluation of the segmentation smoothness
    seq_output_path = os.path.join(os.path.dirname(args.predictions_path), "seq_predictions")
    os.makedirs(seq_output_path, exist_ok=True)
    convert_individual_preds_to_seq(args.predictions_path, args.original_test_path, seq_output_path)