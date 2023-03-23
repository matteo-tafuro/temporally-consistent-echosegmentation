from typing import Tuple, List, Union
from skimage import io
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
import tifffile


def convert_mhd_to_nifti(input_filename: str, output_filename_truncated: str, is_2d_or_3d: str, spacing=None,
                              size=(512,512), is_seg: bool = False) -> None:
    """
    Reads an image in mhd format, resizes it to the desired shape and converts it into a series of niftis.
    The image can have an arbitrary number of input channels which will be exported separately (_0000.nii.gz,
    _0001.nii.gz, etc for images and only .nii.gz for seg).
    Spacing can be ignored most of the time.

    Segmentations will be converted to np.uint32!

    :param is_seg:
    :param transform:
    :param input_filename:
    :param output_filename_truncated: do not use a file ending for this one! Example: output_name='./converted/image1'. This
    function will add the suffix (_0000) and file ending (.nii.gz) for you.
    :param spacing:
    :return:
    """
    
    is_2d_or_3d = is_2d_or_3d.lower()
    assert is_2d_or_3d in ['2d', '3d'], 'is_2d_or_3d must be either `2d` or `3d`'

    img_itk = sitk.ReadImage(input_filename, sitk.sitkFloat64 if not is_seg else sitk.sitkUInt8)
    old_spacing = np.array(img_itk.GetSpacing())
    old_dim = np.array(img_itk.GetSize())
    new_dim = np.array([size[0], size[1], old_dim[2]])

    interp_mode = 'nearest-exact' if is_seg else 'bilinear'

    img = torch.from_numpy(sitk.GetArrayFromImage(img_itk)).unsqueeze(0) if len(img_itk.GetSize()) == 3 else torch.from_numpy(sitk.GetArrayFromImage(img_itk)).unsqueeze(0).unsqueeze(0)
    img = F.interpolate(img, size=size, mode=interp_mode)
    img = img.numpy()
    if is_seg:
            assert img.shape[0] == 1, 'segmentations can only have one color channel, not sure what happened here'

    # Compute new spacing
    if spacing is None:
        spacing = old_spacing * old_dim / new_dim
    if is_2d_or_3d == '2d':
        spacing[2] = 999

    for j, i in enumerate(img):

        if is_seg:
            i = (i).astype(np.uint8)

        itk_img = sitk.GetImageFromArray(i)
        itk_img.SetSpacing(spacing)
        if not is_seg:
            sitk.WriteImage(itk_img, output_filename_truncated + "_%04.0d.nii.gz" % j)
        else:
            sitk.WriteImage(itk_img, output_filename_truncated + ".nii.gz")


def convert_mhd_to_multiple_nifti(input_filename: str, output_filename_truncated: str, is_2d_or_3d: str, spacing=None,
                              size=(512,512), is_seg: bool = False) -> None:
    """
    This function is essentially the same as convert_mhd_to_nifti, but it takes a sequence of N (rather than a single frame) as an
    input and outputs N different images.

    :param is_seg:
    :param transform:
    :param input_filename:
    :param output_filename_truncated: do not use a file ending for this one! Example: output_name='./converted/image1'. This
    function will add the suffix (_0000) and file ending (.nii.gz) for you.
    :param spacing:
    :return:
    """

    is_2d_or_3d = is_2d_or_3d.lower()
    assert is_2d_or_3d in ['2d', '3d'], 'is_2d_or_3d must be either `2d` or `3d`'
    
    img_itk = sitk.ReadImage(input_filename, sitk.sitkFloat64 if not is_seg else sitk.sitkUInt8)
    old_spacing = np.array(img_itk.GetSpacing())
    old_dim = np.array(img_itk.GetSize())
    new_dim = np.array([size[0], size[1], old_dim[2]])

    interp_mode = 'nearest-exact' if is_seg else 'bilinear'

    img = torch.from_numpy(sitk.GetArrayFromImage(img_itk)).unsqueeze(0) if len(img_itk.GetSize()) == 3 else torch.from_numpy(sitk.GetArrayFromImage(img_itk)).unsqueeze(0).unsqueeze(0)
    img = F.interpolate(img, size=size, mode=interp_mode)
    img = img.numpy()
    if is_seg:
        assert img.shape[0] == 1, 'segmentations can only have one color channel, not sure what happened here'

    # Compute new spacing
    if spacing is None:
        spacing = old_spacing * old_dim / new_dim
    if is_2d_or_3d == '2d':
        spacing[2] = 999

    for frame in range(img.shape[1]):

        i = img[:, frame, :, :]

        if is_seg:
            i = (i).astype(np.uint8)

        itk_img = sitk.GetImageFromArray(i)
        itk_img.SetSpacing(spacing)

        frame_filename = f'{output_filename_truncated}_frame_{frame:02d}'

        if not is_seg:
            sitk.WriteImage(itk_img, frame_filename + "_%04.0d.nii.gz" % 0)
        else:
            sitk.WriteImage(itk_img, frame_filename + ".nii.gz")


def convert_3d_tiff_to_nifti(filenames: List[str], output_name: str, spacing: Union[tuple, list], transform=None, is_seg=False) -> None:
    """
    filenames must be a list of strings, each pointing to a separate 3d tiff file. One file per modality. If your data
    only has one imaging modality, simply pass a list with only a single entry

    Files in filenames must be readable with

    Note: we always only pass one file into tifffile.imread, not multiple (even though it supports it). This is because
    I am not familiar enough with this functionality and would like to have control over what happens.

    If Transform is not None it will be applied to the image after loading.

    :param transform:
    :param filenames:
    :param output_name:
    :param spacing:
    :return:
    """
    if is_seg:
        assert len(filenames) == 1

    for j, i in enumerate(filenames):
        img = tifffile.imread(i)

        if transform is not None:
            img = transform(img)

        itk_img = sitk.GetImageFromArray(img)
        itk_img.SetSpacing(list(spacing)[::-1])

        if not is_seg:
            sitk.WriteImage(itk_img, output_name + "_%04.0d.nii.gz" % j)
        else:
            sitk.WriteImage(itk_img, output_name + ".nii.gz")


def convert_2d_segmentation_nifti_to_img(nifti_file: str, output_filename: str, transform=None, export_dtype=np.uint8):
    img = sitk.GetArrayFromImage(sitk.ReadImage(nifti_file))
    assert img.shape[0] == 1, "This function can only export 2D segmentations!"
    img = img[0]
    if transform is not None:
        img = transform(img)

    io.imsave(output_filename, img.astype(export_dtype), check_contrast=False)


def convert_3d_segmentation_nifti_to_tiff(nifti_file: str, output_filename: str, transform=None, export_dtype=np.uint8):
    img = sitk.GetArrayFromImage(sitk.ReadImage(nifti_file))
    assert len(img.shape) == 3, "This function can only export 3D segmentations!"
    if transform is not None:
        img = transform(img)

    tifffile.imsave(output_filename, img.astype(export_dtype))
