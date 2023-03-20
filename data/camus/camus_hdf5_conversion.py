import h5py
import os
import torchio as tio
from tqdm import tqdm
import numpy as np
import pandas as pd

def import_and_resize(file, size, ftype):
    """	
    Read a *.mhd file stored in path, resize it and return it as a numpy array.
    args:
        file: path to the file
        size: tuple of the desired size (W, H)
        ftype: file type, can be 'sequence' or 'mask', used to determie the interpolation method
    returns:
        sequence_resized: numpy array of shape (no_frames, H, W) if the sequence is 3D, or (H, W) otherwise.
    """

    assert ftype in ['sequence', 'mask'], "ftype must be either 'sequence' or 'mask'."

    array = tio.ScalarImage(file).data             # Returns tensor of shape (channels, W, H, no_frames)
    if ftype == 'sequence': array = array / 255.0  # Normalize the sequence to [0,1] if it's a sequence
    array = array.permute(0,2,1,3)                 # Invert H and W to get   (channels, H, W, no_frames)
    array_resized = tio.Resize((size[0], size[1], array.shape[3]), image_interpolation='bspline' if ftype=='sequence' else 'nearest')(array)

    # If the array is 3D (time dimension, i.e. it's the entire sequence) we return it as (no_frames, H, W).
    # Otherwise, it's a single frame or a mask and we return it as (H, W).
    idx_order = (2,0,1) if len(array_resized.squeeze().shape) == 3 else (0,1)

    return array_resized.squeeze().permute(idx_order).numpy() 


def create_hdf5(data_path, out_file_path, size, overwrite=False):
    """
    Create a hdf5 file of the CAMUS training dataset. It will contain the sequence files as data, and the metadata as attributes.
    It will be hierarchically organized as follows:
        |- patient1
            |- 2ch
                |- data: numpy array of shape (no_frames, H, W)
                |- masks: numpy array of shape (2, H, W). First channel is ED mask, second channel is ES mask.
                |- attributes: ED, ES, NbFrame, Sex, Age, ImageQuality, LVedv, LVesv, LVef
            |- 4ch
                |- data
                |- masks
                |- attributes
        |- patient2
            |- 2ch
                |- ...
                |- ...
            |- ... 
        |- ...

    args:
        data_path: path to the folder containing the videos
        out_file_path: path to the output hdf5 file
        size: tuple of the desired size (W, H)
    """

    assert os.path.exists(data_path), f"Folder {data_path} does not exist."

    # Check if file already exists. If so, check the overwrite flag.
    if os.path.exists(out_file_path):
        if overwrite:
            os.remove(out_file_path)
        else:
            print(f'File {out_file_path} already exists. Please set overwrite=True if you want to overwrite it.')
            return

    hdf5 = h5py.File(out_file_path, 'w')
    for root, _, files in tqdm(os.walk(data_path), total=len(os.listdir(data_path))):
        for file in files:
            if file.split('_')[-1] == 'sequence.mhd':
                
                # Get patient and view
                patient, view = file.split('_')[:2]

                # Load video
                sequence = import_and_resize(os.path.join(root, file), size, ftype='sequence')
                chunk_shape = (1, size[0], size[1])
                hdf5.create_dataset(f'{patient}/{view}/data', data=sequence, chunks=chunk_shape, compression="gzip", compression_opts=9)

                # Get path of the two masks
                mask_ED_path = os.path.join(root, f"{patient}_{view}_ED_gt.mhd")
                mask_ES_path = os.path.join(root, f"{patient}_{view}_ES_gt.mhd")
                # Load masks
                mask_ED = import_and_resize(mask_ED_path, size, ftype='mask')
                mask_ES = import_and_resize(mask_ES_path, size, ftype='mask')
                hdf5.create_dataset(f'{patient}/{view}/masks', data=np.stack((mask_ED, mask_ES), axis=0), chunks=chunk_shape, compression="gzip", compression_opts=9)

                # Find info file and store metadata as attributes in hdf5
                info_file_path = os.path.join(root, f"Info_{view}.cfg")
                with open(info_file_path) as f:
                    lines = f.readlines()
                    # ED: X (int)
                    hdf5[f'{patient}/{view}'].attrs.create(lines[0].split(':')[0].strip(), lines[0].split(':')[1].strip(), dtype=int)
                    # ES: X (int)
                    hdf5[f'{patient}/{view}'].attrs.create(lines[1].split(':')[0].strip(), lines[1].split(':')[1].strip(), dtype=int)
                    # NbFrame: X (int)
                    hdf5[f'{patient}/{view}'].attrs.create(lines[2].split(':')[0].strip(), lines[2].split(':')[1].strip(), dtype=int)
                    # Sex: X (str)
                    hdf5[f'{patient}/{view}'].attrs.create(lines[3].split(':')[0].strip(), lines[3].split(':')[1].strip())
                    # Age: X (int)
                    hdf5[f'{patient}/{view}'].attrs.create(lines[4].split(':')[0].strip(), lines[4].split(':')[1].strip(), dtype=int)
                    # ImageQuality: X (str)
                    hdf5[f'{patient}/{view}'].attrs.create(lines[5].split(':')[0].strip(), lines[5].split(':')[1].strip())
                    # LVedv: X (float)
                    hdf5[f'{patient}/{view}'].attrs.create(lines[6].split(':')[0].strip(), lines[6].split(':')[1].strip(), dtype=float)
                    # LVesv: X (float)
                    hdf5[f'{patient}/{view}'].attrs.create(lines[7].split(':')[0].strip(), lines[7].split(':')[1].strip(), dtype=float)
                    # LVef: X (float)
                    hdf5[f'{patient}/{view}'].attrs.create(lines[8].split(':')[0].strip(), lines[8].split(':')[1].strip(), dtype=float)

    hdf5.close()


if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='./training', 
                        help='Path to the training directory')
    parser.add_argument('--out_file_path', type=str, default='./camus_training.hdf5',
                        help='Path to the output hdf5 file')
    parser.add_argument('--size', type=int, nargs=2, default=(512, 512),
                        help='Size of the output images')
    parser.add_argument('--overwrite', default=False, action='store_true', 
                        help='boolean to overwrite the output file if it already exists')
    args = parser.parse_args()

    create_hdf5(args.data_path, args.out_file_path, args.size, args.overwrite)