import torch
from torch import nn, Tensor
from torchir.utils import Resampler
from torchir.transformers import Transformer


class DLIRFramework(nn.Module):
    """Deep Learning Image Registration (DLIR) Framework, a pipeline for transformers.

    The framework is an adaptation of the code presented in `De Vos 2019 <https://www.sciencedirect.com/science/article/pii/S1361841518300495>`_.

    Note: The framework differs from the original paper in the sense that DVFs are not combined by
    addition, but by composition which should result in an increase of registration precision in
    most cases `Hering et al. 2019 <https://link.springer.com/chapter/10.1007/978-3-030-32226-7_29>`_.
    """

    def __init__(self, only_last_trainable: bool = True, return_field: bool = False):
        super().__init__()
        self._resampler = Resampler(coord_dim=-1)
        self.stages = nn.ModuleList()
        self.only_last_trainable = only_last_trainable
        self.return_field = return_field

    def add_stage(self, network: nn.Module, transformer: Transformer):
        if self.only_last_trainable and len(self.stages) > 0:
            self.stages[-1].eval()
            for param in self.stages[-1].parameters():
                param.requires_grad = False
        stage = nn.ModuleDict({"network": network, "transformer": transformer})
        self.stages.append(stage)

    def forward(self, fixed: Tensor, moving: Tensor, return_coordinate_grid: bool = False) -> Tensor:

        # coord_grid_list = []
        total_dvf = None

        coord_grid = None
        warped = moving
        for stage in self.stages:
            parameters = stage["network"](fixed, warped)
            warped, coord_grid, dvf = stage["transformer"](
                parameters, fixed, moving, coord_grid, return_coordinate_grid=True
            )

            total_dvf = dvf if total_dvf is None else total_dvf + dvf

        if return_coordinate_grid:
            return warped, coord_grid, total_dvf
        else:
            return warped

    def apply_to_mask(self, moving_mask: Tensor, coordinate_grid: Tensor = None) -> Tensor:
        with torch.no_grad():
            return self._resampler(moving_mask, coordinate_grid, choose_mode="nearest")
