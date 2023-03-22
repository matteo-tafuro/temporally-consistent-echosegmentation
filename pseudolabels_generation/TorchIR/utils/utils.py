import os
import torch
import h5py
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from models import get_model

def load_model(model_path, mode='eval', device='cpu', verbose=True, map_location=None):
    """
    Load a model from a given path. Set it to evaluation mode if mode='eval'.
    args:
        model_path: path to the model
        mode: 'eval' or 'train'
        device: device to load the model on
    returns:
        model: loaded model in specified mode
    """

    assert mode in ['eval', 'train'], "Mode must be 'eval' or 'train'"

    # Parse text file
    with open(os.path.join(model_path, "model_kwargs.txt"), "r") as f:
        kwargs_lines = f.read().splitlines()
    model_name = kwargs_lines[0]
    kwargs = eval(kwargs_lines[1])
    
    # Init model
    model = get_model(model_name, **kwargs)
    if model_name == 'LitDLIRFramework':
        kwargs_stages = eval(kwargs_lines[2])
        for stage in kwargs_stages:
            model.add_stage(get_model(**stage).net, get_model(**stage).transformer)
    
    # Load checkpoint
    if map_location=='cpu':
        state_dict = torch.load(os.path.join(model_path, "checkpoints", "best_model_iter.pt"), map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(os.path.join(model_path, "checkpoints", "best_model_iter.pt"))
    model.load_state_dict(state_dict)
    
    if map_location != 'cpu':
        model.to(device)
    
    if verbose: print(f"Checkpoint of {model_name} correctly loaded (on device: {device}) and set to evaluation mode.")
    
    if mode == 'eval':
        return model.eval()
    return model

def get_frames_masks(patient, view, file=None, filepath=None):
    """
    Retrieve frames and masks from a HDF5 file.
    For the HDF5 file, can either pass the file object or the filepath. At least one is required.
    args:
        patient: patient number (either an int with no leading zeros, or a string in the form `patientXXXX`)
        view: view name (2CH or 4CH)
        filepath: path to the HDF5 file
    returns:
        frames: frames of the patien twith shape [no_frames, H, W]
        mask_first: mask at ED with shape [1, 1, H, W]
        mask_last: mask at ES with shape [1, 1, H, W]
    """

    assert file is not None or filepath is not None, "Either file or filepath must be provided"

    # Import HDF5 if not already done
    if file is None:
        file = h5py.File(filepath, 'r')
    # Retrieve frames
    patient = 'patient' + (str(patient).zfill(4)) if type(patient) == int else patient
    frames = file[f'{patient}/{view}/data']
    # Retrieve masks (first element is mask at ED, second mask is ES)
    mask_first, mask_last = file[f'{patient}/{view}/masks']
    # Convert to tensors
    mask_first = torch.from_numpy(mask_first).unsqueeze(0).unsqueeze(0)
    mask_last = torch.from_numpy(mask_last).unsqueeze(0).unsqueeze(0)
    
    if file is None:
        file.close()

    return frames, mask_first, mask_last