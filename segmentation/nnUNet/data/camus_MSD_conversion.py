import os
import re
import pandas as pd
from tqdm import tqdm
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data
from nnunet.utilities.file_conversions import convert_mhd_to_nifti, convert_mhd_to_multiple_nifti

if __name__ == '__main__':
    """
    nnU-Net was originally built for 3D images. It is also strongest when applied to 3D segmentation problems because a 
    large proportion of its design choices were built with 3D in mind. Also note that many 2D segmentation problems, 
    especially in the non-biomedical domain, may benefit from pretrained network architectures which nnU-Net does not
    support.
    Still, there is certainly a need for an out of the box segmentation solution for 2D segmentation problems. And 
    also on 2D segmentation tasks nnU-Net cam perform extremely well! We have, for example, won a 2D task in the cell 
    tracking challenge with nnU-Net (see our Nature Methods paper) and we have also successfully applied nnU-Net to 
    histopathological segmentation problems. 
    Working with 2D data in nnU-Net requires a small workaround in the creation of the dataset. Essentially, all images 
    must be converted to pseudo 3D images (so an image with shape (X, Y) needs to be converted to an image with shape 
    (1, X, Y). The resulting image must be saved in nifti format. Hereby it is important to set the spacing of the 
    first axis (the one with shape 1) to a value larger than the others. If you are working with niftis anyways, then 
    doing this should be easy for you. This example here is intended for demonstrating how nnU-Net can be used with 
    'regular' 2D images. We selected the massachusetts road segmentation dataset for this because it can be obtained 
    easily, it comes with a good amount of training cases but is still not too large to be difficult to handle.
    """

    train = True
    test = True

    train_dense = True
    test_dense = True
    test_both = True

    ted_only = False
    if ted_only:
        ted2camus = pd.read_csv('../../../data/ted2camus.csv')
        patients_of_interest = ted2camus['camus_id'].tolist()

    is_2d_or_3d = '2d'

    # test_both cant be True if test_dense is False
    assert not (test_both and not test_dense)

    # download data and unzip files, then set the following path according to your system:
    base = '../../datasets/camus'

    assert os.path.isdir(base), "Please set the path to the dataset in this script"

    # now start the conversion to nnU-Net:
    task_name = 'Task570_2D_CAMUS_sparse'
    target_base = os.path.join(nnUNet_raw_data, task_name) # nnUNet_raw_data is a env variable that should be set

    target_imagesTr = os.path.join(target_base, "imagesTr")
    target_imagesTs = os.path.join(target_base, "2D_TestSet")
    target_imagesTs_seq = os.path.join(target_base, "2D_TestSet_Sequences")
    target_labelsTr = os.path.join(target_base, "labelsTr")

    os.makedirs(target_imagesTr, exist_ok=True)
    os.makedirs(target_imagesTs, exist_ok=True)
    os.makedirs(target_imagesTs_seq, exist_ok=True)
    os.makedirs(target_labelsTr, exist_ok=True)

    training_dir = os.path.join(base, 'training')
    test_dir = os.path.join(base, 'testing')

    # Define resize size for images
    size = (512, 512)

    # =============================  TRAINING ON SPARSE DATASET  =================================

    if train:
        if not train_dense:
            for root, _, files in tqdm(os.walk(training_dir), total=len(os.listdir(training_dir)), desc="Training (sparse) samples..."):
                for file in files:
                    if file.endswith("_gt.mhd") and ("sequence" not in file):
                        # Define inputs
                        input_segmentation_file = os.path.join(root, file)
                        input_image_file = input_segmentation_file.replace("_gt", "")

                        # Retrieve unique filename
                        unique_name = os.path.basename(input_image_file)[:-4]
                        # Define outputs
                        output_image_file = os.path.join(target_imagesTr,
                                        unique_name)  # do not specify a file ending! This will be done for you
                        output_seg_file = os.path.join(target_labelsTr, unique_name)  # do not specify a file ending! This will be done for you
        
                        # this utility will convert 2d images that can be read by skimage.io.imread to nifti. You don't need to do anything.
                        # if this throws an error for your images, please just look at the code for this function and adapt it to your needs
                        convert_mhd_to_nifti(input_image_file, output_image_file, is_seg=False, size=size, is_2d_or_3d=is_2d_or_3d)
                        convert_mhd_to_nifti(input_segmentation_file, output_seg_file, is_seg=True, size=size, is_2d_or_3d=is_2d_or_3d)

    # =============================  TRAINING ON DENSE DATASET  =================================

        else: # if train_dense:
            for root, _, files in tqdm(os.walk(training_dir), total=len(os.listdir(training_dir)), desc="Training (dense) samples..."):
                for file in files:
                    if file.endswith("_sequence_gt.mhd"):
                        # Define inputs
                        input_segmentation_file = os.path.join(root, file)
                        input_image_file = input_segmentation_file.replace("_gt", "")

                        # Retrieve unique filename
                        unique_name = os.path.basename(input_image_file)[:-4]
                        # Define outputs
                        output_image_file = os.path.join(target_imagesTr,
                                        unique_name)  # do not specify a file ending! This will be done for you
                        output_seg_file = os.path.join(target_labelsTr, unique_name)  # do not specify a file ending! This will be done for you
        
                        # this utility will convert 2d images that can be read by skimage.io.imread to nifti. You don't need to do anything.
                        # if this throws an error for your images, please just look at the code for this function and adapt it to your needs
                        convert_mhd_to_multiple_nifti(input_image_file, output_image_file, is_seg=False, size=size, is_2d_or_3d=is_2d_or_3d)
                        convert_mhd_to_multiple_nifti(input_segmentation_file, output_seg_file, is_seg=True, size=size, is_2d_or_3d=is_2d_or_3d)

    # # =============================  TESTING  =================================
    
    if test:
        if (not test_dense) or test_both:
            for root, _, files in tqdm(os.walk(test_dir), total=len(os.listdir(test_dir)), desc="Test samples..."):
                for file in files:
                    pattern = re.compile(r'[24]CH_E[SD].mhd')
                    if pattern.search(file):
                        # Define inputs
                        input_image_file = os.path.join(root, file)
                        # Retrieve unique filename
                        unique_name = os.path.basename(input_image_file)[:-4]
                        # Define outputs
                        output_image_file = os.path.join(target_imagesTs,
                                        unique_name)  # do not specify a file ending! This will be done for you
        
                        # this utility will convert 2d images that can be read by skimage.io.imread to nifti. You don't need to do anything.
                        # if this throws an error for your images, please just look at the code for this function and adapt it to your needs
                        convert_mhd_to_nifti(input_image_file, output_image_file, is_seg=False, size=size, is_2d_or_3d=is_2d_or_3d)

    # =============================  TESTING ON A FULL SEQUENCE  =================================
    
        if test_dense or test_both:
            for root, _, files in tqdm(os.walk(test_dir), total=len(os.listdir(test_dir)), desc="Test samples for ..."):
                for file in files:
                    if file.endswith("_sequence.mhd"):
                        if ted_only and ((int(file.split("_")[0][-4:]) not in patients_of_interest) or ("4CH" not in file)):
                            continue
                        # Define inputs
                        input_image_file = os.path.join(root, file)
                        # Retrieve unique filename
                        unique_name = os.path.basename(input_image_file)[:-4]
                        # Define outputs
                        output_image_file = os.path.join(target_imagesTs_seq,
                                        unique_name)  # do not specify a file ending! This will be done for you
        
                        # this utility will convert 2d images that can be read by skimage.io.imread to nifti. You don't need to do anything.
                        # if this throws an error for your images, please just look at the code for this function and adapt it to your needs
                        convert_mhd_to_multiple_nifti(input_image_file, output_image_file, is_seg=False, size=size, is_2d_or_3d=is_2d_or_3d)

    # finally we can call the utility for generating a dataset.json
    generate_dataset_json(os.path.join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('Ultrasound',),
                          labels={0: 'background', 1: 'LV_endo', 2: 'LV_epi', 3: 'LA'}, dataset_name=task_name, license='hands off!')

    """
    once this is completed, you can use the dataset like any other nnU-Net dataset. Note that since this is a 2D
    dataset there is no need to run preprocessing for 3D U-Nets. You should therefore run the 
    `nnUNet_plan_and_preprocess` command like this:
    > nnUNet_plan_and_preprocess -t 120 -pl3d None
    once that is completed, you can run the trainings as follows:
    > nnUNet_train 2d nnUNetTrainerV2 120 FOLD
    (where fold is again 0, 1, 2, 3 and 4 - 5-fold cross validation)
    there is no need to run nnUNet_find_best_configuration because there is only one model to choose from.
    Note that without running nnUNet_find_best_configuration, nnU-Net will not have determined a postprocessing
    for the whole cross-validation. Spoiler: it will determine not to run postprocessing anyways. If you are using
    a different 2D dataset, you can make nnU-Net determine the postprocessing by using the
    `nnUNet_determine_postprocessing` command
    """