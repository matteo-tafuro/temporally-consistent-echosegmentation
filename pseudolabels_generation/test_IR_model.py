import numpy as np
from tqdm.notebook import tqdm
import os
import sys
# Custom imports
# Custom imports
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(MAIN_DIR))
from TorchIR.utils.utils import get_frames_masks, load_model
from TorchIR.utils.plots import generate_video, plot_masks_comparison, jacobian_std_boxplot, average_warped_image
from propagate_masks import propagate_masks, propagate_masks_bidirectional


POOR_2CH = ('2CH', 271, 'Poor')
MEDIUM_2CH = ('2CH', 437, 'Medium')
GOOD_2CH = ('2CH', 178, 'Good')
POOR_4CH = ('4CH', 13, 'Poor')
MEDIUM_4CH = ('4CH', 82, 'Medium')
GOOD_4CH = ('4CH', 359, 'Good')

cases = [POOR_2CH, MEDIUM_2CH, GOOD_2CH, POOR_4CH, MEDIUM_4CH, GOOD_4CH]


def generate_test_plots(model_path, hdf5_filepath, device='cpu', verbose=False):

    # Load model
    model = load_model(model_path, mode='eval', device=device, verbose=verbose)
    
    # create output directory if it does not exist
    out_path = os.path.join(model_path, 'eval')
    os.makedirs(out_path, exist_ok=True)

    # Initialize dict to store the STD of the Jacobian determinant
    jac_std = {}

    pbar = tqdm(cases)
    for (view, patient, quality) in pbar:
        pbar.set_description(f"ID {patient} {view} ({quality.capitalize()} video quality)", refresh=True)
        
        # Load data
        frames, mask_first, mask_last = get_frames_masks(patient, view, filepath=hdf5_filepath)

        # Propagate masks
        if verbose: print("Propagating segmentation masks...")
        warped_masks, dvfs, (jacobians, jacobian_masked) = propagate_masks(frames, mask_first, model=model, device=device, verbose=False)
        
        # create a masked numpy array of warped_masks along the first dimension and hide the zeros
        warped_masks_masked = np.ma.masked_where(warped_masks == 0, warped_masks)

        # Store STD of the Jacobian determinant for the boxplots
        jac_std[f'{quality.upper()}_{view}'] = jacobians[1:].std(axis=(1,2)) # skip the first jacobian mask because it has no motion

        # Generate and save the video 
        if verbose: print("Generating an animation of the propagation...")
        generate_video(frames, warped_masks_masked, out_path, quality, view, patient, jacobians, jacobian_masked)
        
        # Save the comparison between the GT and the last mask
        if verbose: print("Saving the comparison between the GT and the last mask...")
        plot_masks_comparison(mask_last.squeeze().cpu().numpy(), warped_masks[-1], out_path, quality, view, patient)
        
        # Generate the average warped masks for qualitative evaluation
        if verbose: print("Generating the average warped masks for qualitative evaluation...")
        average_warped_image(model, frames, out_path, quality, view, patient, device=device)

    if verbose: print("Generating boxplot of the Jacobian's STD...")
    # Boxplot with the jacobian std of the different patients
    jacobian_std_boxplot(jac_std, out_path)


def generate_test_plots_bidirectional(model_path, hdf5_filepath='/home/mtarufo/thesis/supervision_by_registration/load_datasets/camus/camus_training.hdf5', device='cpu', verbose=False):

    # Load model
    model = load_model(model_path, mode='eval', device=device, verbose=verbose)
    
    # create output directory if it does not exist
    out_path = os.path.join(model_path, 'eval_bidirectional')
    os.makedirs(out_path, exist_ok=True)

    pbar = tqdm(cases)
    for (view, patient, quality) in pbar:
        pbar.set_description(f"ID {patient} {view} ({quality.capitalize()} video quality)", refresh=True)
        
        # Load data
        frames, mask_first, mask_last = get_frames_masks(patient, view, hdf5_filepath)

        # Propagate masks
        if verbose: print("Propagating segmentation masks...")
        warped_masks = propagate_masks_bidirectional(frames, mask_first, mask_last, model=model, device=device, verbose=False)
        
        # create a masked numpy array of warped_masks along the first dimension and hide the zeros
        warped_masks_masked = np.ma.masked_where(warped_masks == 0, warped_masks)

        # Generate and save the video 
        if verbose: print("Generating an animation of the propagation...")
        generate_video(frames, warped_masks_masked, out_path, quality, view, patient)
        