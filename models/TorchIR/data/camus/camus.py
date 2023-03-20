import os
import numpy as np
from itertools import permutations
import pandas as pd
import h5py
import torch
import torchio as tio
from torch.utils.data import Dataset
import pandas as pd


class CamusSequenceDataset(Dataset):
    """
    Dataloader for the unsupervised training on the Camus dataset based on the hdf5 file created by `data_preparation.py`.
    """
    def __init__(self, file_path, transform=None, leave_out_patients=25, ted2camus_path=None, test_set=False):

        assert os.path.exists(file_path), f"File {file_path} does not exist. Please run the `data_preparation.py` script in the /camus directory to create it."

        # Assert that if leave_out_patient is -1, ted2camus_path is not None and it is a valid path of an existing file
        if leave_out_patients == -1:
            assert ted2camus_path is not None, "ted2camus_path must be specified if leave_out_patients is -1"
            assert os.path.exists(ted2camus_path), f"File {ted2camus_path} does not exist."

        super().__init__()
        self.file = None
        self.file_path = file_path
        self.transform = transform
        self.leave_out_patients = leave_out_patients
        self.test_set = test_set
        self.df = pd.DataFrame(columns=['patient', 'id', 'view', 'frame_no'])

        df_frames_dir = os.path.join(os.path.dirname(self.file_path), 'lookup_tables')
        df_frames_path = os.path.join(df_frames_dir, f"df_frames_{leave_out_patients if leave_out_patients>0 else 'leave_out_ted'}.csv")

        # Extract the patients to leave out.
        if self.leave_out_patients >= 0:
            self.leave_out_indices = np.arange(1, self.leave_out_patients+1)
        else:
            # check that the file exists. change path below
            ted2camus = pd.read_csv(ted2camus_path)
            self.leave_out_indices = list(ted2camus["camus_id"])

        create = True
        # If the dataframe already exists (with the correct amount of leave_out_patients) we load it
        if not test_set and os.path.exists(df_frames_path):	
            self.df = pd.read_csv(df_frames_path)
            create = False

        # Otherwise we create it.
        # Because an opened HDF5 file isn’t pickleable and to send Dataset to workers’ processes it needs to be serialised with pickle, you can’t 
        # open the HDF5 file in __init__. Open it in __getitem__ and store as the singleton!. Do not open it each time as it introduces huge overhead.
        # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643
        # However, we temporarily open the file to generate the dataframe with all frames of all patients
        with h5py.File(self.file_path, 'r') as file:
            if create:

                if not os.path.exists(df_frames_dir):
                    os.makedirs(df_frames_dir)

                print(f"CSV file could not be found at path `{df_frames_path}`. Generating it now. This may take a while...")
                for patient in file.keys():
                        patient_id = int(patient[-4:])
                        # If we're using the Dataloader for the test set we only consider the first `leave_out_patients` patients, 
                        # otherwise we discard the first `leave_out_patients` patients  
                        if (test_set) and (leave_out_patients != 0) and (patient_id not in self.leave_out_indices): continue       # if testing
                        if (not test_set) and (leave_out_patients != 0) and (patient_id in self.leave_out_indices): continue       # if training
        
                        for view in file[patient].keys():
                            n_frames = file[f'{patient}/{view}'].attrs['NbFrame']

                            sample_seq = pd.DataFrame([[patient, patient_id, view]]*n_frames, columns=['patient', 'id', 'view'])
                            sample_seq['frame_no'] = np.arange(1,n_frames+1)
                            self.df = pd.concat([self.df, sample_seq], ignore_index=True)

                # Save csv           
                if not test_set:
                    self.df.to_csv(df_frames_path, index = False)
                    
        self.no_unique_patients = len(set(self.df['patient']))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx >= len(self): raise IndexError # Preventing out of bounds, as seen here https://stackoverflow.com/questions/54640906/torch-dataset-looping-too-far
       
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')

        patient, patient_id, view, frame_no = list(self.df.loc[idx])

        # Load image with shape (W,H)
        image = self.file[f'{patient}/{view}/data'][frame_no-1]

        if self.transform is not None:
            # Transformations are implemented using Albumentations
            image = self.transform(image=image)['image']

        # We return the image as a 3D tensor with shape (1, H, W) to be compatible with the rest of the code
        return torch.from_numpy(image).unsqueeze(0), patient_id, frame_no

    def select_transforms(self, dataset):
        # Define whether we are using the training or validation set, to correctly select the desired transfromations.
        self.dataset = dataset

class IRCamusSet(CamusSequenceDataset):
    """
    Wrapper to convert the Camus Dataset into a dataset suitable for image registration experiments. 
    We will generate every possible permutation of frames *belonging to the same patient* (and having
    the same view).
    """

    def __init__(self, file_path, *args, **kwargs):
        super().__init__(file_path, *args, **kwargs)
        self.df_permutations = pd.DataFrame(columns=['patient', 'id', 'view', 'fixed_frame', 'moving_frame'])

        df_permutation_dir = os.path.join(os.path.dirname(self.file_path), f'lookup_tables')
        df_permutations_path = os.path.join(df_permutation_dir, f"df_permutations_{self.leave_out_patients if self.leave_out_patients>0 else 'leave_out_ted'}.csv")

        create = True
        if os.path.exists(df_permutations_path):	
                self.df_permutations = pd.read_csv(df_permutations_path)
                create = False

        if create:
            print(f"CSV file could not be found at path `{df_permutations_path}`. Generating it now. This may take a while...")

            if not os.path.exists(df_permutation_dir):
                os.makedirs(df_permutation_dir)

            for patient in set(self.df['patient']):
                for view in ['2CH', '4CH']:
                    frames = self.df[(self.df['patient'] == patient) & (self.df['view'] == view)]['frame_no'].to_numpy()
                    permut = np.array(list(permutations(frames, 2)))
                    fixed_frames = permut[:,0]
                    moving_frames = permut[:,1]
                    data = {'patient': patient,
                        'id': int(patient[-4:]),
                        'view': view,
                        'fixed_frame': fixed_frames,
                        'moving_frame': moving_frames}
                    self.df_permutations = pd.concat([self.df_permutations, pd.DataFrame(data)], ignore_index=True)
            # Save csv
            self.df_permutations.to_csv(df_permutations_path, index = False)

    def __len__(self):
        return len(self.df_permutations)

    def __getitem__(self, idx):

        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')
            
        patient, _, view, fixed_frame_idx, moving_frame_idx = list(self.df_permutations.loc[int(idx)])
        
        # read images
        fixed_image =  self.file[f'{patient}/{view}/data'][fixed_frame_idx-1]
        moving_image = self.file[f'{patient}/{view}/data'][moving_frame_idx-1]
        
        if self.transform is not None:
            # Transformations are implemented using Albumentations
            fixed_image, moving_image = self.transform(image=fixed_image)['image'], self.transform(image=moving_image)['image']

        # We return the image as a 3D tensor with shape (1, H, W) to be compatible with the rest of the code
        return {"fixed": torch.from_numpy(fixed_image).unsqueeze(0), "moving": torch.from_numpy(moving_image).unsqueeze(0)}

    def select_transforms(self, dataset):
        # Define whether we are using the training or validation set, to correctly select the desired transfromations.
        self.dataset = dataset

class CamusSubSet(CamusSequenceDataset):
    """
    A Dataset class that selects all frames of a single patient.
    """
    def __init__(self, patient_id, view, *args, **kwargs):
        super().__init__(leave_out_patients=0, *args, **kwargs)

        assert(patient_id >= 0 and patient_id <= self.no_unique_patients)
        assert(view in ['2CH', '4CH'])
        
        self.df = self.df[(self.df['id'] == patient_id) & (self.df['view'] == view)].reset_index(drop=True)
        self.no_unique_patients = 1