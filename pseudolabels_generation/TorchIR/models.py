# Standard libraries and Pytorch
import os
import torch.nn as nn

# Custom imports
from torchir.networks import DIRNet, AIRNet
from torchir.transformers import BsplineTransformer, AffineTransformer
from torchir.dlir_framework import DLIRFramework

def get_model(model, **kwargs):
    assert model in ["LitDIRNet", "LitAIRNet", "LitDLIRFramework"], "Model must be either 'LitDIRNet', 'LitAIRNet' or 'LitDLIRFramework'"
    
    if model == "LitDIRNet":
        return LitDIRNet(**kwargs)
    elif model == "LitAIRNet":
        return LitAIRNet(**kwargs)
    elif model == "LitDLIRFramework":
        return LitDLIRFramework(**kwargs)

# ========================================================= LitDIRNet =========================================================

class LitDIRNet(nn.Module):
    def __init__(self,
        grid_spacing,       # for DIRNet
        upsampling_factors, # for BsplineTransformer
        # DIRNet parameters
        kernel_size=3,
        kernels=32,
        num_conv_layers=5,
        num_dense_layers=2,
        ndim_dirnet=2,
        # BsplineTransformer parameters
        ndim_bspline=2,
):

        """
        PyTorch module that summarizes all components to train a single DIRNet.
        Inputs:
            grid_spacing - Spacing of the grid used for the BsplineTransformer
            kernels - Number of kernels used in the DIRNet
        """
        super().__init__()
        self.net = DIRNet(
            grid_spacing=grid_spacing, 
            kernels=kernels,
            kernel_size = kernel_size,
            num_conv_layers=num_conv_layers,
            num_dense_layers=num_dense_layers,
            ndim = ndim_dirnet)

        self.transformer = BsplineTransformer(
            ndim=ndim_bspline,
            upsampling_factors=upsampling_factors)

    def forward(self, fixed, moving, return_coordinate_grid=False):
        """
        The forward function calculates 
        Inputs:
            fixed - Fixed image.
            moving - Moving image.
        Ouptuts:
            warped - Warped moving image.
        """

        params = self.net(fixed, moving)
        if return_coordinate_grid:
            warped, coordinate_grid, dvf = self.transformer(params, fixed, moving, return_coordinate_grid=return_coordinate_grid)
            return warped, coordinate_grid, dvf
        else: 
            return self.transformer(params, fixed, moving)

    @property
    def device(self):
        """
        Property function to get the device on which the model is.
        """
        return self.net.device

    @property
    def name(self):
        """
        Property function to get the name of the model.
        """
        return "DIRNet"

# ========================================================= LitAIRNet =========================================================

class LitAIRNet(nn.Module):
    def __init__(self, 
        # AIRNet parameters
        kernel_size=3,
        kernels=32,
        linear_nodes=64,
        num_conv_layers=5,
        num_dense_layers=2,
        num_downsamplings=4,
        ndim_airnet=2,
        # AffineTransformer parameters
        ndim_affine=2
        ):

        """
        PyTorch module that summarizes all components to train a single AIRnet.
        Inputs:
            kernels - Number of kernels used in the AIRNet

        """
        super().__init__()
        self.net = AIRNet(
            kernel_size=kernel_size,
            kernels=kernels,
            linear_nodes=linear_nodes,
            num_conv_layers=num_conv_layers,
            num_dense_layers=num_dense_layers,
            num_downsamplings=num_downsamplings,
            ndim=ndim_airnet)

        self.transformer = AffineTransformer(
            ndim=ndim_affine)

    def forward(self, fixed, moving):
        """
        The forward function calculates 
        Inputs:
            fixed - Fixed image.
            moving - Moving image.
        Ouptuts:
            warped - Warped moving image.
        """

        params = self.net(fixed, moving)
        warped  = self.transformer(params, fixed, moving)
        return warped

    @property
    def device(self):
        """
        Property function to get the device on which the model is.
        """
        return self.net.device

    @property
    def name(self):
        """
        Property function to get the name of the model.
        """
        return "AIRNet"

# ========================================================= DLIRFramework =========================================================

class LitDLIRFramework(nn.Module):
    def __init__(self, only_last_trainable=True):

        """
        PyTorch module for the DLIRFramework, which uses coarse-to-fine image registration. This is a dynamic implementation,
        where we add a layer and we train it. We add another layer, fix the weights of the previous layer and train the new one etc.
        Inputs:
            only_last_trainable - Boolean that indicates if only the last layer is trainable or all of them.
        """
        super().__init__()
        self.net = DLIRFramework(only_last_trainable=only_last_trainable)
        self.add_stage = self.net.add_stage

    def forward(self, fixed, moving, return_coordinate_grid=False):
        """
        The forward function calculates 
        Inputs:
            fixed - Fixed image.
            moving - Moving image.
        Ouptuts:
            warped - Warped moving image.
        """
        if return_coordinate_grid:
            warped, coordinate_grid, dvf_list = self.net(fixed, moving, return_coordinate_grid=return_coordinate_grid)
            return warped, coordinate_grid, dvf_list
        else:
            return self.net(fixed, moving, return_coordinate_grid=return_coordinate_grid)

    @property
    def device(self):
        """
        Property function to get the device on which the model is.
        """
        return self.net.device

    @property
    def name(self):
        """
        Property function to get the name of the model.
        """
        return "DLIRFramework"


# ========================================================= ENSEMBLE =========================================================

