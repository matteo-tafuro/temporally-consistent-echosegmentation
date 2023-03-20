import torch
from .camus import IRCamusSet
import os

def load_train_val_with_transforms(file_path, transforms, leave_out_patients, val_set_size, verbose = True):
    """
    Load and split dataset into train and validation set using the specified val set size. Use the specified transforms for train and validation set.
    Inspired by https://discuss.pytorch.org/t/changing-transforms-after-creating-a-dataset/64929/8
    args:
        file_path: path to the dataset in HDF5 format
        transforms: dictionary containing the transforms for train and validation set
        leave_out_patients: number of patients to leave out for evaluation
        val_set_size: size of the validation set
        verbose: print additional information
    returns:
        train_dataset: training set
        val_dataset: validation set
    """

    ted2camus_path = None
    if leave_out_patients == -1:
        ted2camus_path = os.path.join(os.path.dirname(file_path), 'ted2camus.csv')

    # Load dataset
    train_dataset = IRCamusSet(file_path=file_path, leave_out_patients=leave_out_patients, ted2camus_path=ted2camus_path, transform=transforms['train'])
    val_dataset =   IRCamusSet(file_path=file_path, leave_out_patients=leave_out_patients, ted2camus_path=ted2camus_path, transform=transforms['val'])
    if verbose: print(f"   -- Loaded frames from {train_dataset.no_unique_patients} patients (the first {leave_out_patients} were left out for evaluation)." )
    
    # Split dataset into train and validation set
    indices = torch.randperm(len(train_dataset), generator=torch.Generator())
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-val_set_size])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[-val_set_size:])

    if verbose: print(f"   -- Created a training set of {len(train_dataset)} image pairs and a validation set of {len(val_dataset)}." )
    
    return train_dataset, val_dataset