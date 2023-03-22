# add all missing imports
import numpy as np
import pandas as pd
import torch
import cv2
import os
import h5py
from tqdm import tqdm
import SimpleITK as sitk

from TorchIR.utils.utils import load_model
from TorchIR.utils.eval import batch_compute_jacobian
from TorchIR.utils.utils import get_frames_masks


def generate_inbetween_frames(data_dir, hdf5_filepath, model_dir, output_dir = None, direction='bidirectional', method='sequential', interested_patients=None, device='cuda'):
    """	
    Generate inbetween frames for each patient using the models in models_dir. The inbetween frames are generated using the approach specified
    by the `direction` argument. If `output_dir` is None, the masks will be saved with the original data. Otherwise, the masks will be saved
    in the `output_dir` directory. The arg `interested_patient` accepts a list of tuples, containing an integer corresponding to the patient
    ID and a view, e.g. [(1, '2CH'), (14, '4CH')]. If given, the inbetween masks will only be generated for those patients.
    The HDF5 is used to speed up the loading of the data.
    args:
        data_dir (str): path to the directory containing the data
        hdf5_filepath (str): path to the HDF5 file containing the data
        model_dir (str): path to the directory containing the model
        output_dir (str): path to the directory where the masks will be saved. If None, the masks will be saved with the original data.
        direction (str): 'bidirectional', 'forward' or 'backward'
        method (str): Method to use for propagation. Can be either 'sequential' or 'fixed'.
        interested_patients (list): list of tuples. If given, the inbetween masks will only be generated for those patients.
        device (str): 'cuda' or 'cpu'
    """

    assert direction in ['bidirectional', 'forward', 'backward'], "Direction must be 'bidirectional', 'forward' or 'backward'."

    # Create output folder if it doesn't exist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = load_model(model_dir, mode='eval', device=device, verbose=False, map_location='cpu')
    model = model.to(device)

    # Initialize HDF5 file
    with h5py.File(hdf5_filepath, 'r') as hdf5_file:
        pbar = tqdm(os.walk(data_dir), total=450 if interested_patients is None else len(interested_patients), desc=f'Generating inbetween frames...')
        for root, _, files in pbar:
            for file in files:
                if file.split('_')[-1] == 'sequence.mhd':
                    
                    # Get patient and view
                    patient, view = file.split('_')[:2]
                    patient_id = int(patient[-4:])
                    pbar.set_description(f'Generating inbetween frames for {patient} ({view})...')

                    # Define output path so that we can check if the file already exists
                    outpath = os.path.join(root, f'{patient}_{view}_sequence_gt_{direction}.mhd') if output_dir is None \
                                else os.path.join(output_dir, patient, f'{patient}_{view}_sequence_gt_{direction}.mhd')
                    
                    # If interested_patients is given, skip the patient if it's not in the list
                    if interested_patients is not None:
                        if (patient_id, view) not in interested_patients:
                            continue
                        # If the file already exists, skip it
                        if os.path.exists(outpath):
                            pbar.update(1)	
                            continue
                    
                    # Store info about the image that we need later when exporting the masks
                    original_img_itk = sitk.ReadImage(os.path.join(root, file), sitk.sitkFloat64)
                    old_spacing = np.array(original_img_itk.GetSpacing())
                    old_dim = np.array(original_img_itk.GetSize())
                    new_dim = np.array([512, 512, old_dim[2]]) # (512, 512) hardcoded for now
                    new_spacing = old_spacing * old_dim / new_dim

                    # Load data
                    frames, mask_first, mask_last = get_frames_masks(patient, view, hdf5_file)

                    # Initialize masks container
                    ref_masks = np.zeros(frames.shape)

                    # Generate inbetween frames ============================================================================================================
                    if direction=='forward':
                        ref_masks = propagate_masks_unidirectional(frames, mask_first, model=model, method=method, device=device, verbose=False)
                    elif direction=='backward':
                        # invert the direction of the frames
                        ref_masks_inverted = propagate_masks_unidirectional(np.flip(frames[:], axis=0).copy(), mask_last, method=method, model=model, device=device, verbose=False)
                        # re-invert the direction of the masks
                        ref_masks = np.flip(ref_masks_inverted, axis=0)
                    elif direction=='bidirectional':
                        ref_masks = propagate_masks_bidirectional(frames, mask_first, mask_last, model=model, method=method, device=device, verbose=False)
                    # Add leading dimension to ref_masks
                    ref_masks = np.expand_dims(ref_masks, axis=0)

                    avg_masks_0 = np.mean(ref_masks == 0, axis=0)
                    avg_masks_1 = np.mean(ref_masks == 1, axis=0)
                    avg_masks_2 = np.mean(ref_masks == 2, axis=0)
                    avg_masks_3 = np.mean(ref_masks == 3, axis=0)

                    # stack all the average masks along the first axis,
                    # then take the argmax along the first axis to get the most common mask
                    avg_masks = np.stack([avg_masks_0, avg_masks_1, avg_masks_2, avg_masks_3], axis=0)
                    avg_masks = np.argmax(avg_masks, axis=0)

                    # =======================================================================================================================================

                    # Save inbetween frames
                    itk_img = sitk.GetImageFromArray(avg_masks)
                    itk_img.SetSpacing(new_spacing)
                    # create outpath folder if it doesn't exist
                    os.makedirs(os.path.dirname(outpath), exist_ok=True)
                    sitk.WriteImage(itk_img, outpath)
                    pbar.update(1)
    
    pbar.close()


def interpolate_masks(warped_masks_forward, warped_masks_backward):
    """
    Aggregate the forward and backward warped masks using a distance transform, as seen in https://stackoverflow.com/questions/51409818/how-to-average-two-masks.
    Args:
        warped_masks_forward: Forward warped masks of shape [T,H,W]
        warped_masks_backward: Backward warped masks of shape [T,H,W]
    Returns:
        warped_interpolated_masks: Interpolated masks of shape [T,H,W]
    """

    # Initialize output
    warped_interpolated_masks = np.zeros_like(warped_masks_forward)
    # Initialize kernel to be used for closing
    kernel = np.ones((5, 5), np.uint8)
    # Initialize weights
    t = np.linspace(0, np.pi/2, warped_masks_forward.shape[0])
    
    for i, (frame_mask_forward, frame_mask_backward) in enumerate(zip(warped_masks_forward, warped_masks_backward)):
        # Initialize per-frame mask container
        mask = np.zeros((4, warped_masks_forward.shape[1], warped_masks_forward.shape[2]))
        
        # Loop over the different classes
        for j in [0,1,2,3]:

            # Extract binary mask 
            binary_forward = (frame_mask_forward == j)
            binary_backward = (frame_mask_backward == j)

            # Compute distance transform
            d1 = cv2.distanceTransform(binary_forward.astype(np.uint8), cv2.DIST_L2, 3) - cv2.distanceTransform((~binary_forward).astype(np.uint8), cv2.DIST_L2, 3)
            d2 = cv2.distanceTransform(binary_backward.astype(np.uint8), cv2.DIST_L2, 3) - cv2.distanceTransform((~binary_backward).astype(np.uint8), cv2.DIST_L2, 3)
            mask[j] = (d1 * np.cos(t[i])**2 + d2 * np.sin(t[i])**2) > 0
        
        # Merge individual masks
        merged_mask = np.argmax(mask, axis=0).astype(np.uint8)
        # Close holes and gaps in the masks using a morphological operation
        warped_interpolated_masks[i] = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, kernel)

    return warped_interpolated_masks

def propagate_masks_bidirectional(frames, first_mask, last_mask, model_path = None, model = None, method =  "sequential", verbose = True, device = 'cpu'):
    """
    Propagate masks using a bidirectional approach, i.e. from the first to the last frame, and from the last to the first frame.
    Then, the class-wise masks are interpolated using a distance transform and eventually combined into a single multi-label mask.
    The function can accept a model path or a model object.
    Args:
        frames: Numpy array of shape [no_frames,H,W] containing the frames
        first_mask: Numpy array of shape [1,1,H,W] containing the mask in the first frame (ED)
        last_mask: Numpy array of shape [1,1,H,W] containing the mask in the last frame (ES)
        model_path: Path to the trained model
        model: Trained model object
        method: Method to use for mask propagation (sequential or fixed)
        verbose: Verbosity
        device: Device to use for inference
    Returns:
        warped_interpolated_masks: Interpolated masks of shape [T,H,W]
    """

    assert model_path is not None or model is not None, "Either `model_path` or `model` must be provided"
    assert method in ["sequential", "fixed"], "Method must be either `sequential` or `fixed`"

    # Load the model
    if model is None:
        model = load_model(model_path, mode='eval', device=device, verbose=verbose)
    else:
        model = model

    # Initialize mask container (forward direction)
    warped_masks_forward = np.zeros_like(frames)
    moving_mask_forward = (first_mask / 255.0).to(device)
    warped_masks_forward[0] = moving_mask_forward.squeeze().cpu().numpy()

    # Initialize mask container (backward direction)
    warped_masks_backward = np.zeros_like(frames)
    moving_mask_backward = (last_mask / 255.0).to(device)
    warped_masks_backward[0] = moving_mask_backward.squeeze().cpu().numpy()

    # Initialize transformer
    transformer = model.transformer if model.name == 'DIRNet' else model.net.stages[0]["transformer"]

    # Loop over each pair of frames
    with torch.no_grad():

        # Generate batches
        if method == "sequential":
            forward_batch = np.lib.stride_tricks.sliding_window_view(frames, (2, frames.shape[1], frames.shape[2]), writeable=True).squeeze()
            backward_batch = np.lib.stride_tricks.sliding_window_view(np.flip(frames[:], axis=0).copy(), (2, frames.shape[1], frames.shape[2]), writeable=True).squeeze()
        else:
            forward_batch = np.array([[frames[0], x] for x in frames[1:]])
            backward_batch = np.array([[np.flip(frames[:], axis=0).copy()[0], x] for x in np.flip(frames[:], axis=0).copy()[1:]])

        forward_moving_batch = torch.from_numpy(forward_batch[:,0,:,:]).unsqueeze(1).to(device)
        forward_fixed_batch = torch.from_numpy(forward_batch[:,1,:,:]).unsqueeze(1).to(device)
        
        backward_moving_batch = torch.from_numpy(backward_batch[:,0,:,:]).unsqueeze(1).to(device)
        backward_fixed_batch = torch.from_numpy(backward_batch[:,1,:,:]).unsqueeze(1).to(device)

        # Run inference
        _, batch_forward_coords, _ = model(forward_fixed_batch, forward_moving_batch, True)
        _, batch_backward_coords, _ = model(backward_fixed_batch, backward_moving_batch, True)

        # To compute the next mask, we need the previous one. This can't be vectorized so we loop over the batch
        for i in np.arange(frames.shape[0]-1):
            # Store and update the mask (forward direction)
            mask_forward = transformer.apply_to_mask(moving_mask_forward, batch_forward_coords[i].unsqueeze(0))
            warped_masks_forward[i+1] = mask_forward.squeeze().cpu().numpy()

            # Store and update the mask (backward direction)
            mask_backward = transformer.apply_to_mask(moving_mask_backward, batch_backward_coords[i].unsqueeze(0))
            warped_masks_backward[i+1] = mask_backward.squeeze().cpu().numpy()

            if method == "sequential":
                moving_mask_forward = mask_forward
                moving_mask_backward = mask_backward

        # Convert to uint8 and create a masked numpy array of warped_masks along the first dimension and hide the zeros
        warped_masks_forward = (warped_masks_forward * 255).astype(np.uint8)
        warped_masks_backward = (warped_masks_backward * 255).astype(np.uint8)


    warped_masks_backward = np.flip(warped_masks_backward, axis=0)
    return interpolate_masks(warped_masks_forward, warped_masks_backward)

def propagate_masks_unidirectional(frames, mask, model_path = None, model = None, method =  "sequential", verbose = True, device = 'cpu'):
    """
    Propagate masks from the first frame to the last frame using a trained registration model.
    The function can accept a model path or a model object.
    Args:
        frames: Numpy array of shape [no_frames,H,W] containing the frames
        mask: Numpy array of shape [1,1,H,W] containing the mask
        model_path: Path to the trained model
        model: Trained model object
        method: Method to use for propagation. Can be either 'sequential' or 'fixed'.
        verbose: Verbosity
        device: Device to use for inference
    Returns:
        warped_masks: Numpy array of shape [no_frames,H,W] containing the propagated masks
        warped_masks_masked: Same as warped_masks but with zeros hidden
        dvfs: Displacement vector fields of shape [no_frames,2,H,W]
    """

    assert model_path is not None or model is not None, "Either model_path or model must be provided"
    assert method in ["sequential", "fixed"], "Method must be either 'sequential' or 'fixed'"

    # Load the model
    if model is None:
        model = load_model(model_path, mode='eval', device=device, verbose=verbose)
    else:
        model = model

    # Initialize masks container
    warped_masks = np.zeros_like(frames)
    moving_mask = (mask / 255.0).to(device)
    warped_masks[0] = moving_mask.squeeze().cpu().numpy()

    # Initialize transformer
    transformer = model.transformer if model.name == 'DIRNet' else model.net.stages[0]["transformer"]

    if verbose: print("Propagating masks...")
    # Loop over each pair of frames
    with torch.no_grad():
        # Generate a batch of consequent frames
        if method == "sequential":
            batch = np.lib.stride_tricks.sliding_window_view(frames, (2, frames.shape[1], frames.shape[2]), writeable=True).squeeze()
        else:
            batch = np.array([[frames[0], x] for x in frames[1:]])
        moving_batch = torch.from_numpy(batch[:,0,:,:]).unsqueeze(1).to(device)
        fixed_batch = torch.from_numpy(batch[:,1,:,:]).unsqueeze(1).to(device)
        # Forward pass
        _, batch_coords, _ = model(fixed_batch, moving_batch, True)

        # To compute the next mask, we need the previous one. This can't be vectorized so we loop over the batch
        for i, coords in enumerate(batch_coords):
            # Store and update the mask (if sequential)
            coords = coords.unsqueeze(0)
            warped_mask = transformer.apply_to_mask(moving_mask, coords)
            warped_masks[i+1] = warped_mask.squeeze().cpu().numpy()
            if method == "sequential":
                moving_mask = warped_mask
        if verbose: print("Done.")

        # Convert to uint8
        warped_masks = (warped_masks * 255).astype(np.uint8)

    return warped_masks

def propagate_masks(frames, mask, model_path = None, model = None, verbose = True, device = 'cpu'):
    """
    Propagate masks from the first frame to the last frame using a trained registration model.
    The function can accept a model path or a model object.
    Args:
        frames: Numpy array of shape [no_frames,H,W] containing the frames
        mask: Numpy array of shape [1,1,H,W] containing the mask
        model_path: Path to the trained model
        model: Trained model object
    Returns:
        warped_masks: Numpy array of shape [no_frames,H,W] containing the propagated masks
        warped_masks_masked: Same as warped_masks but with zeros hidden
        dvfs: Displacement vector fields of shape [no_frames,2,H,W]
    """

    assert model_path is not None or model is not None, "Either model_path or model must be provided"

    # Load the model
    if model is None:
        model = load_model(model_path, mode='eval', device=device, verbose=verbose)
    else:
        model = model

    # Initialize masks container
    warped_masks = np.zeros_like(frames)
    moving_mask = (mask / 255.0).to(device)
    warped_masks[0] = moving_mask.squeeze().cpu().numpy()

    # Initialize DVF container
    dvfs = np.zeros((len(frames), 2, *frames.shape[1:]))

    # Initialize transformer
    transformer = model.transformer if model.name == 'DIRNet' else model.net.stages[0]["transformer"]

    if verbose: print("Propagating masks...")
    # Loop over each pair of frames
    with torch.no_grad():
        # Generate a batch of consequent frames
        batch = np.lib.stride_tricks.sliding_window_view(frames, (2, frames.shape[1], frames.shape[2]), writeable=True).squeeze()
        moving_batch = torch.from_numpy(batch[:,0,:,:]).unsqueeze(1).to(device)
        fixed_batch = torch.from_numpy(batch[:,1,:,:]).unsqueeze(1).to(device)
        # Forward pass
        _, batch_coords, dvfs = model(fixed_batch, moving_batch, True)

        # To compute the next mask, we need the previous one. This can't be vectorized so we loop over the batch
        for i, coords in enumerate(batch_coords):
            # Store and update the mask
            coords = coords.unsqueeze(0)
            warped_mask = transformer.apply_to_mask(moving_mask, coords)
            warped_masks[i+1] = warped_mask.squeeze().cpu().numpy()
            moving_mask = warped_mask
        if verbose: print("Done.")

        # Convert to uint8
        warped_masks = (warped_masks * 255).astype(np.uint8)
        # Place an empty mask in the first index position
        dvfs = dvfs.cpu().numpy()
        dvfs = np.insert(dvfs, 0, np.zeros_like(dvfs[0]), axis=0)
        # Compute the jacobian of the displacement vector fields
        if verbose: print("Computing Jacobian determinants...")
        jacobians, jacobian_masked = batch_compute_jacobian(dvfs)
        if verbose: print("Done.")

    return warped_masks, dvfs, (jacobians, jacobian_masked)


if __name__ == '__main__':
    
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    method = 'sequential'
    direction = 'bidirectional'
    only_generated_TED = False
    
    data_dir = '../data/camus/training'
    hdf5_filepath = '../data/camus/camus_training.hdf5'
    model_dir = 'TorchIR/output/LitDIRNet/leave_out_TED'

    # Let's only generate the inbetween frames for the patients in the TED set, if specified so
    if only_generated_TED:
        ted2camus_df = pd.read_csv('../data/ted2camus.csv')
        interested_patients = [(ID, '4CH') for ID in ted2camus_df['camus_id']]

    output_dir = f'../data/propagated_masks/{method}/{direction}'
    generate_inbetween_frames(data_dir, hdf5_filepath, model_dir, direction=direction, output_dir = output_dir, interested_patients=interested_patients if only_generated_TED else None, method=method, device=device)