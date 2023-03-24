import os
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from eval import compute_area_label

def convert_individual_preds_to_seq(predictions_path, original_test_path, output_path):
    """
    Convert individual predictions to sequences of predictions. A sequence is generated for each unique patient in `predictions_path`.
    The outputs are saved in `output_path`.
    args:
        predictions_path: Path to the folder containing the predictions
        original_test_path: Path to the original test set
        output_path: Path where the sequences will be saved
    """

    # Create output folder
    os.makedirs(output_path, exist_ok=True)

    files = [file for file in os.listdir(predictions_path) if file.endswith(".nii.gz")]
    unique_patients = list(set([file.split("_")[0] for file in files]))

    for patient in unique_patients:
        # find files in files that start with patient
        patient_all_files = [file for file in files if file.startswith(patient)]
        # find view
        for view in ['2CH', '4CH']:
            patient_view_files = [file for file in patient_all_files if view in file]
            sorted_files = sorted(patient_view_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            # put the whole sequence in a 3D array first. then resize and convert to itk
            sequence = np.zeros((len(sorted_files), 512, 512))
            for i, file in enumerate(sorted_files):
                sequence[i] = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(predictions_path, file)))
            
            # import the same sequence from the original to get the spacing
            original_sequence = sitk.ReadImage(os.path.join(original_test_path, patient, f'{patient}_{view}_sequence.mhd'))
            original_spacing = original_sequence.GetSpacing()

            # Set spacing to the newly created sequence, then save it
            sequence_itk = sitk.GetImageFromArray(sequence)
            sequence_itk.SetSpacing(original_spacing)
            # print(sequence_itk.GetSize(), sequence_itk.GetSpacing())
            sitk.WriteImage(sequence_itk, os.path.join(output_path, f'{patient}_{view}_sequence.mhd'))

def retrieve_gt_area(patient, view, root = 'data/camus/testing/'):
    """
    Retrieve the area of the ground truth for a given patient and view.
    args:
        patient: Patient ID
        view: View ID
        root: Path to a specific split of the dataset (e.g. `data/camus/training/` or `data/camus/testing/`)
    returns:
        A list of lists, where each sublist contains the area of the ground truth for each label in the sequence.
    """

    output = []

    ED_path = os.path.join(root, patient, f"{patient}_{view}_ED_gt.mhd")
    ES_path = os.path.join(root, patient, f"{patient}_{view}_ES_gt.mhd")
    filepaths = [ED_path, ES_path]

    for path in filepaths:
        # Read ground truth file
        img_itk = sitk.ReadImage(path)
        img = torch.from_numpy(sitk.GetArrayFromImage(img_itk)).unsqueeze(0)
        img = F.interpolate(img, size=(512,512), mode='nearest-exact')
        img = img.squeeze().numpy()
        output.append([compute_area_label(img, label) for label in [0, 1, 2, 3]])

    return output


def retrieve_ED_ES_frames(patient, view, keep_original_ratio = False, root = 'data/camus/testing/'):
    """
    Retrieve the frame and the corresponding mask at ED and ES for a given patient and view.
    args:
        patient: Patient ID
        view: View ID
        keep_original_ratio: If True, the images are not resized to (512,512)
        root: Path to a specific split of the dataset (e.g. `data/camus/training/` or `data/camus/testing/`)
    returns:	
        ED: List containing the frame and the corresponding mask at ED
        ES: List containing the frame and the corresponding mask at ES
        ED_size: Size of the frame at ED
        ES_size: Size of the frame at ES
    """
    
    ED = []
    ES = []
    
    ED_file = os.path.join(root, patient, f"{patient}_{view}_ED.mhd")
    ED_mask = os.path.join(root, patient, f"{patient}_{view}_ED_gt.mhd")
    ED_filepaths = [ED_file, ED_mask]

    ES_file = os.path.join(root, patient, f"{patient}_{view}_ES.mhd")
    ES_mask = os.path.join(root, patient, f"{patient}_{view}_ES_gt.mhd")
    ES_filepaths = [ES_file, ES_mask]

    # ED 
    for path in ED_filepaths:
        # Read ground truth file
        img_itk = sitk.ReadImage(path)
        ED_size = img_itk.GetSize()[:2]
        ED_spacing = img_itk.GetSpacing()
        ED_aspect_ratio = ED_spacing[0] / ED_spacing[1] 
        img = torch.from_numpy(sitk.GetArrayFromImage(img_itk).astype(np.float32)).unsqueeze(0) / 255.0
        if not keep_original_ratio:
            img = F.interpolate(img, size=(512,512), mode='nearest-exact' if 'gt' in path else 'bilinear')
        img = img.squeeze().numpy()
        if 'gt' in path:
            img = (img * 255).astype(np.uint8) 
        ED.append(img)

    # ES
    for path in ES_filepaths:
        # Read ground truth file
        img_itk = sitk.ReadImage(path)
        ES_spacing = img_itk.GetSpacing()
        ES_aspect_ratio = ES_spacing[0] / ES_spacing[1]
        ES_size = img_itk.GetSize()[:2]
        img = torch.from_numpy(sitk.GetArrayFromImage(img_itk).astype(np.float32)).unsqueeze(0) / 255.0
        if not keep_original_ratio:
            img = F.interpolate(img, size=(512,512), mode='nearest-exact' if 'gt' in path else 'bilinear')
        img = img.squeeze().numpy()
        if 'gt' in path:
            img = (img * 255).astype(np.uint8) 

        ES.append(img)

    return ED, ES, ED_size[::-1], ES_size[::-1]


def retrieve_ED_ES_preds(patient, view, model, seq_path, ED_size=None, ES_size=None):
    """
    Retrieve the frame and the corresponding mask at ED and ES for a given patient and view, from the predictions of a given model.
    If the size of the frames at ED and ES are provided, the images are resized using the specified size.
    args:
        patient: Patient ID
        view: View ID
        model: the name of the model, as saved in the `seq_path` folder
        seq_path: Path to the folder containing the predictions of the models
        ED_size: Size of the frame at ED
        ES_size: Size of the frame at ES
    returns:
        ED: predicted mask at ED
        ES: predicted at ES
    """

    model_path = os.path.join(seq_path, model)
    file = os.path.join(model_path, f"{patient}_{view}_sequence.mhd")
    img_itk = sitk.ReadImage(file)
    img = torch.from_numpy(sitk.GetArrayFromImage(img_itk).astype(np.float32)).unsqueeze(0) / 255.0
    
    img_ED, img_ES = img[:,0,...].unsqueeze(0), img[:,-1,...].unsqueeze(0)
    if ED_size is not None and ES_size is not None:
        img_ED = F.interpolate(img_ED, size=(ED_size), mode='nearest-exact')
        img_ES = F.interpolate(img_ES, size=(ES_size), mode='nearest-exact')
    img_ED = (img_ED.squeeze().numpy() * 255).astype(np.uint8)
    img_ES = (img_ES.squeeze().numpy() * 255).astype(np.uint8)
    return img_ED, img_ES