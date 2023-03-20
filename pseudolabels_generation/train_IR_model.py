# Standard libraries
import argparse
import os
import datetime
import random
import numpy as np
from tqdm import tqdm
import sys

# PyTorch
import torch

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, 'TorchIR'))
# Custom imports
from TorchIR.models import get_model
from TorchIR.dataloader import load_train_val_with_transforms
from TorchIR.torchir.metrics import NCC
from test_IR_model import generate_test_plots

# Path to the output of the training
DEST_DIR = os.path.join(SCRIPT_DIR, 'TorchIR', 'output')

# =============================================== VARIOUS FUNCTIONS ===============================================

def seed_everything(seed):
    """
    Function to set the seed for all random number generators.
    Inputs:
        seed - Seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def train_model(model, data_batch, optimizer, loss_module, mask, bending_penalty, device):
    """
    Train the model on the provided data loader.
    Inputs:
        model - PyTorch model
        data_loader - PyTorch DataLoader
        optimizer - PyTorch optimizer
        loss_module - PyTorch loss module
        mask - mask for the loss function
        bending_penalty - bool, whether to use bending penalty
    Outputs:
        loss - Loss value of the model on the provided data
    """

    ## Step 1: Move input data to device
    fixed_batch = data_batch['fixed']
    moving_batch = data_batch['moving']
    fixed_batch, moving_batch = fixed_batch.to(device), moving_batch.to(device)

    ## Step 2: Run the model on the input data
    warped_batch, batch_grid, _ = model(fixed_batch, moving_batch, return_coordinate_grid=True)

    ## Step 3: Calculate the loss
    if mask is not None:
        train_loss = loss_module(fixed_batch, warped_batch, mask, batch_grid=batch_grid if bending_penalty else None)
    else:
        train_loss = loss_module(fixed_batch, warped_batch, batch_grid=batch_grid if bending_penalty else None)

    ## Step 4: Perform backpropagation
    optimizer.zero_grad()
    train_loss.backward()

    ## Step 5: Update the parameters
    optimizer.step()

    return train_loss.item()
    

def test_model(model, data_loader, loss_module, mask, bending_penalty, device):
    """
    Test the model on the provided data loader.
    Inputs:
        model - PyTorch model
        data_loader - PyTorch DataLoader
        loss_module - PyTorch loss module
        mask - mask for the loss function
        bending_penalty - bool, whether to use bending penalty
    Outputs:
        loss - Loss value of the model on the provided data
    """
    model.eval()
    average_loss = 0
    with torch.no_grad():
        val_iterator = tqdm(data_loader, desc="Validation...", leave=False)
        for batch in val_iterator:
            # Move data to device
            fixed_batch = batch['fixed']
            moving_batch = batch['moving']
            fixed_batch, moving_batch = fixed_batch.to(device), moving_batch.to(device)
            # Forward pass
            warped_batch, batch_grid, _ = model(fixed_batch, moving_batch, return_coordinate_grid=True)
            # Calculate the loss
            if mask is not None:
                batch_loss = loss_module(fixed_batch, warped_batch, mask, batch_grid=batch_grid if bending_penalty else None)
            else:
                batch_loss = loss_module(fixed_batch, warped_batch, batch_grid=batch_grid if bending_penalty else None)
            # Calculate the running average loss
            average_loss += batch_loss.item()/len(val_iterator)

    return average_loss

# =============================================== MAIN FUNCTION ===============================================

def main(args):
    """
    Inputs:
        args - Namespace object from the argument parser
    """

    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    print('Using device:', device)

    if args.seed is not None:
        seed_everything(args.seed)

    assert args.model in ["single", "affine", "hierarchical"], 'Model must be one of "single", "affine", or "hierarchical".'
    assert args.augment_prob >= 0 and args.augment_prob <= 1, 'Augmentation probability must be between 0 and 1.'

    model_name = "LitDIRNet" if args.model == "single" else "LitAIRNet" if args.model == "affine" else "LitDLIRFramework"

    # Transforms can be added here to augment the training set
    train_transform = None

    # Define the transforms
    transformations = {
        'train': train_transform if args.augment else None,
        'val': None,
    }

    # Prepare logging
    if args.save: 
        base_dir = os.path.join(DEST_DIR, args.output_dir) if args.output_dir is not None else DEST_DIR
        model_dir = os.path.join(base_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        experiment_dir = os.path.join(
            model_dir, 
            (f"{'seed_'+str(args.seed)+'__' if args.seed is not None else ''}"
            f"{'BPE_'+'_'.join(str(args.alpha).split('.'))+'__' if args.bp else ''}"                   # Bending penalty
            f"{'masked__' if args.mask_loss else ''}"                                                   # Masked loss                         
            f"{'no_augment__' if not args.augment else f'augment__'}"                                        # Augmentation
            f"{'' if not args.augment else 'p'+'_'.join(str(args.augment_prob).split('.'))+'__'}"       # Augmentation probability
            f"{datetime.datetime.now().strftime('date_%d_%m_%Y__time_%H_%M_%S')}")                      # Date and time
        )
        os.makedirs(experiment_dir, exist_ok=True)
        checkpoint_dir = os.path.join(
            experiment_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        writer = SummaryWriter(experiment_dir)

    print("== Loading data...")

    # Load dataset
    train_set, val_set = load_train_val_with_transforms(file_path=args.data_dir, transforms=transformations, leave_out_patients=args.leave_out_patients, val_set_size=400)

    loss_mask = None
    if args.mask_loss:
        loss_mask = train_set[0]['fixed'].squeeze() > 1e-2
        loss_mask = loss_mask.to(device).unsqueeze(0).unsqueeze(0)

    # Define DataLoaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, persistent_workers=True if args.num_workers > 0 else False, num_workers=args.num_workers, pin_memory=True)

    print("== Initializing model and optimizer...")

    # Create model
    stages = [None] # Only used for hierarchical model
    if args.model == "single":
        kwargs = {"grid_spacing": (32, 32), "kernels": 32, "upsampling_factors": (32,32)}
    elif args.model == "affine":
        kwargs = {"kernels": 16}
    elif args.model == "hierarchical":
        kwargs = {"only_last_trainable": True}
        kwargs_stages = [
                         {"model": "LitDIRNet", "grid_spacing":(32, 32), "kernels": 32, "upsampling_factors":(32,32)},
                         {"model": "LitDIRNet", "grid_spacing":(16, 16), "kernels": 16, "upsampling_factors":(16,16)},
                         {"model": "LitDIRNet", "grid_spacing":( 8,  8), "kernels":  8, "upsampling_factors": (8, 8)  },
                        #  {"model": "LitDIRNet", "grid_spacing":( 4,  4), "kernels":  4, "upsampling_factors": (4, 4)  }
                         ]
        stages = [get_model(**kwargs_stages[i]) for i in range(len(kwargs_stages))] 
    
    # Save model info in a text file
    if args.save: 
        with open(os.path.join(experiment_dir, "model_kwargs.txt"), 'w') as f:
            f.write(f"{model_name}\n{str(kwargs)}\n{f'{str(kwargs_stages)}' if args.model=='hierarchical' else ''}")

    model = get_model(model_name, **kwargs)

    # If hierarchical was chosen, add first stage
    if args.model == "hierarchical":
        args.iterations *= len(stages)
        stage_progress = 0
        model.add_stage(stages[stage_progress].net, stages[stage_progress].transformer)
    model = model.to(device)

    # Create optimizer and loss module
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.net.parameters()), lr=args.lr, amsgrad=True)
    loss_module = NCC(use_mask=args.mask_loss, bending_penalty= args.bp, alpha=args.alpha)

    # Tracking variables for finding best model
    best_val_ncc = float('inf')
    stage_iter = 0
    tot_iter = 0
    best_iter_idx = tot_iter

    train_loss_array = np.zeros((args.iterations // args.log_every_n_steps) + 1)
    val_loss_array = np.zeros((args.iterations // args.val_check_interval) + 1)

    tmp = f"== Training a {model_name} loop with the following settings:"
    for arg in vars(args):
        tmp += f"\n   -- {arg}: {getattr(args, arg)}"
    print(tmp)
    if args.save:
        with open(os.path.join(experiment_dir, "setup.txt"), 'w') as f:
            f.write(tmp)
    
    # Initialize progress bar
    pbar = tqdm(
        total=args.iterations,
        desc=f"Training {f'(Stage {stage_progress+1})' if len(stages)>1 else ''}...")
    pbar.update(1)

    # Create infinite-looping dataloader, as seen in https://discuss.pytorch.org/t/infinite-dataloader/17903/6
    data_iter = iter(train_loader) 

    for i in range(args.iterations):
        # Get next batch
        try:
            data_batch = next(data_iter)
        # StopIteration is thrown if dataset ends. In that case, reinitialize data loader
        except StopIteration:
            data_iter = iter(train_loader)
            data_batch = next(data_iter)
        
        model.train()

        ##############
        #  TRAINING  #
        ##############
        train_loss = train_model(model, data_batch, optimizer, loss_module, loss_mask, args.bp, device=device)
    
        # Log training loss
        if stage_iter % args.log_every_n_steps == 0 or stage_iter == 1:
            last_train_iter = stage_iter
            train_loss_array[tot_iter // args.log_every_n_steps] = train_loss
            if args.save:
                writer.add_scalar('training_NCC', train_loss, global_step = tot_iter)

        ##############
        # VALIDATION #
        ##############
        if stage_iter % args.val_check_interval == 0 or stage_iter == 1:
            last_val_iter = stage_iter
            # Validate model on validation set
            val_loss = test_model(model, val_loader, loss_module, loss_mask, args.bp, device=device)
            # Log validation loss
            val_loss_array[tot_iter // args.val_check_interval] = val_loss
            if args.save:
                writer.add_scalar('validation_NCC', val_loss, global_step = tot_iter)
                # Saving best model
                if val_loss < best_val_ncc:
                    best_val_ncc = val_loss
                    best_iter_idx = tot_iter
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best_model_iter.pt"))
        
        # Update progress counters
        stage_iter += 1
        tot_iter += 1
        
        # If model is hierarchical, add next stage and reset stage counter
        if args.model == "hierarchical" and stage_iter == args.iterations//len(stages) and stage_progress < len(stages) - 1:
            stage_iter = 1
            stage_progress += 1
            print(f'Global iter {tot_iter}, or iter {stage_iter} in stage {stage_progress}.')
            print(f"  -- Proceeding to stage {stage_progress+1}...")
            model.add_stage(stages[stage_progress].net.to(device), stages[stage_progress].transformer.to(device))
            # Re-initialize the optimizer with the new params
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.net.parameters()), lr=args.lr, amsgrad=True)
            best_val_ncc = float('inf')

        # Update progress bar
        pbar.update(1)
        pbar.set_description(f"Training {f'(Stage {stage_progress+1})' if len(stages)>1 else ''}... |\
                                Train loss ({last_train_iter}): {train_loss_array[last_train_iter // args.log_every_n_steps]:.3f} |\
                                Val loss ({last_val_iter}): {val_loss_array[last_val_iter // args.val_check_interval]:.3f}")

    if args.save:
        writer.close()
    pbar.close()
    
    # Test the model
    if args.test and args.save:
        print("Testing model...")
        generate_test_plots(experiment_dir, hdf5_filepath=args.data_dir, device=device, verbose=False)


# =============================================== ARG PARSING ===============================================

if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--model', default="single", type=str,
                        help='What model to use. Can be "single", "affine" or "hierarchical".')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Minibatch size')

    # Dataset
    parser.add_argument('--data_dir', default=r"../data/camus/camus_training.hdf5", type=str, 
                        help='Directory where to look for the data.')
    parser.add_argument('--leave_out_patients', default='25', type=int,
                        help='How many patients to leave out from the train/val set (for testing). If -1, then the patients \
                            from the TED project are left out. Note: the script assumes that the `ted2camus.csv` file is in the \
                            `/data` folder.')

    # Data augmentation
    parser.add_argument('--augment', action='store_true',
                        help='Whether to apply data augmentation or not')
    parser.add_argument('--no_augment', dest='augment', action='store_false',
                        help='Whether to avoid applying data augmentation or not')
    parser.set_defaults(augment=False)
    parser.add_argument('-p', '--augment_prob', default=0.5, type=float,
                        help='Probability of applying data augmentation to a given sample')
                        
    # Other hyperparameters
    parser.add_argument('--gpu', default=0, type=int,
                        help='Which GPU to use')
    parser.add_argument('-i', '--iterations', default=10000, type=int,
                        help='Max number of iterations')
    parser.add_argument('--log_every_n_steps', default=100, type=int,
                         help='How often to log training progress')
    parser.add_argument('--val_check_interval', default=100, type=int,
                        help='How often to run validation')
    parser.add_argument('--seed', default=2022, type=int,
                        help='Seed to use for reproducing results. If -1, no seed is used.')
    parser.add_argument('-n', '--num_workers', default=24, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0. ')
    
    # Whether to save the models
    parser.add_argument('--save', action='store_true',
                        help='Whether to save the outputs of the training.')
    parser.add_argument('--no_save', dest='save', action='store_false',
                        help='Whether to avoid saving the outputs of the training.')
    parser.set_defaults(save=True)
    parser.add_argument('--output_dir', default=None, type=str, 
                        help='Directory where to save the outputs. The directory will be created within the `output` folder.')

    # Whether to use a mask for the loss
    parser.add_argument('--mask_loss', action='store_true',
                        help='Whether to use a mask for the loss.')
    parser.add_argument('--no_mask_loss', dest='mask_loss', action='store_false',
                        help='Whether to avoid using a mask for the loss.') 
    parser.set_defaults(mask_loss=False)

    # Whether to use the bending penalty
    parser.add_argument('-a', '--alpha', default=1, type=float,
                        help='Hyperparameter for the bending energy penalty.')
    parser.add_argument('--bp', action='store_true',
                        help='Whether to include the bending energy penalty in the loss function.')
    parser.add_argument('--no_bp', dest='bp', action='store_false',
                        help='Whether to avoid including the bending energy penalty in the loss function.') 
    parser.set_defaults(bp=True)

    # Whether to test the models and generate all plots
    parser.add_argument('--test', action='store_true',
                        help='Whether to test the models and generate all plots.')
    parser.add_argument('--no_test', dest='test', action='store_false',
                        help='Whether to avoid testing the the models and generate all plots.') 
    parser.set_defaults(test=True)

    # Parse arguments
    args = parser.parse_args()
    
    main(args)

