import json
import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from tqdm import tqdm


def compute_geometric_metrics(mask, gt, label, spacing=1, smoothing=1e-6):
    """
    Compute the DICE score, mean surface distance, root mean square surface distance,
    Hausdorff distance between two binary images, and the area of the label in the predicted segmentation
    
    args:
        mask          The segmentation that has been created
        gt            The GT segmentation against which we wish to compare mask
        label         The label of the structure of interest
        spacing       The pixel resolution or pixel size. This is entered as an n-vector where n is equal to the number of dimensions in the segmentation i.e. 2D or 3D.
                      The default value is 1 which means pixels (or rather voxels) are 1 x 1 x 1 mm in size.
        smoothing     A small number to prevent division by zero
    returns:
        dice          The DICE score
        msd           The mean surface distance
        rms           The root mean square surface distance
        hd            The Hausdorff distance
        area          The area of the label
    """

    # DICE score
    dice = compute_dice(mask, gt, label, smoothing)

    # Surface distance
    surface_distance = compute_surface_dist(mask, gt, label, sampling=spacing)
    
    # Surface distance-based metrics
    msd = surface_distance.mean()                # Mean surface distance
    rms = np.sqrt((surface_distance**2).mean())  # Root mean square surface distance
    hd  = surface_distance.max()                 # Hausdorff distance

    # Compute area of label
    area = compute_area_label(mask, label)

    return dice, msd, rms, hd, area

def compute_surface_dist(input1, input2, label, sampling=1, connectivity=1):
    """
    Compute the surface distance between two binary images, as seen https://mlnotebook.github.io/post/surface-distance-function/
    args:
        input1        The segmentation that has been created
        input2        The GT segmentation against which we wish to compare input1
        sampling      The pixel resolution or pixel size. This is entered as an n-vector where n is equal to the number of dimensions in the segmentation i.e. 2D or 3D.
                      The default value is 1 which means pixels (or rather voxels) are 1 x 1 x 1 mm in size.
        connectivity  Creates either a 2D (3 x 3) or 3D (3 x 3 x 3) matrix defining the neighbourhood around which the function looks for neighbouring pixels.
                      Typically, this is defined as a six-neighbour kernel which is the default behaviour of this function.
    returns:
        sds           The surface distance between the two binary images
    """

    input_1 = np.atleast_1d((input1 == label))
    input_2 = np.atleast_1d((input2 == label))

    conn = generate_binary_structure(input_1.ndim, connectivity)

    S = (input_1.astype(np.float32) - binary_erosion(input_1, conn).astype(np.float32)).astype(bool)
    Sprime = (input_2.astype(np.float32) - binary_erosion(input_2, conn).astype(np.float32)).astype(bool)

    dta = distance_transform_edt(~S,sampling)
    dtb = distance_transform_edt(~Sprime,sampling)
    
    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])
       
    return sds

def compute_dice(mask, gt, label, smoothing=1e-6):
    return (np.sum((mask == label)[gt == label]) * 2.0 + smoothing) / (np.sum(mask == label) + np.sum(gt == label) + smoothing)

def compute_area_label(mask, label):
    return np.sum(mask == label)

def compute_jacobian_determinant(dvf_np):
    """
    Compute the jacobian determinant of a 2D displacement field
    Args:
        dvf_np: Displacement field of shape [2,H,W]
    Returns:
        jacobian: Jacobian determinant of shape [H,W]
        jacobian_masked: Masked jacobian determinant of shape [H,W] where the values are less than zero
    """
    #Convert the numpy array to a HxW image with each pixel being a 2D vector and compute the jacobian determinant volume
    sitk_displacement_field = sitk.GetImageFromArray(dvf_np.transpose(1,2,0), isVector=True)
    jacobian_det_volume = sitk.DisplacementFieldJacobianDeterminant(sitk_displacement_field)
    jacobian = sitk.GetArrayViewFromImage(jacobian_det_volume,).astype(np.float16)
    # Find the critical points, i.e. where the jacobian determinant is less than zero
    jacobian_masked = np.ma.masked_where((jacobian > 0), jacobian)

    return jacobian, jacobian_masked

def batch_compute_jacobian(dvfs):
    """
    Compute the jacobian determinant of a batch of displacement fields of shape [B,2,H,W]
    Args:
        dvfs: Displacement fields of shape [B,2,H,W]
    Returns:
        jacobians: Jacobian determinants of shape [B,H,W]
        jacobian_masked: Masked jacobian determinants of shape [B,H,W] where the values are less than zero
    """

    # TODO: Vectorize

    jacobians = np.zeros((dvfs.shape[0], dvfs.shape[2], dvfs.shape[3]))
    jacobian_masked = np.ma.zeros((dvfs.shape[0], dvfs.shape[2], dvfs.shape[3]))

    for i in range(dvfs.shape[0]):
        jacobians[i], jacobian_masked[i] = compute_jacobian_determinant(dvfs[i])

    return jacobians, jacobian_masked


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
