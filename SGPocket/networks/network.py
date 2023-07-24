from typing import List

import torch
import torch.nn.functional as F
from torch_geometric.nn.models import MLP

from SGPocket.networks.spherical_convolution import SphericalConvolution_PYG


class SGCN(torch.nn.Module):
    """Spherical harmonic Graph Convolutional Network
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels_list: List[int],
                 mlp_channels_list: List[int],
                 dropout: float,
                 order: int,
                 non_linearity: str):
        """Constructor

        Args:
            in_channels (int): Input channel
            hidden_channels_list (List[int]): Hidden channels
            mlp_channels_list (List[int]): MLP channels
            dropout (float): Dropout
            order (int): Spherical harmonics order
            non_linearity (str): Non linearity
        """
        super().__init__()
        self.conv_list = torch.nn.ModuleList([SphericalConvolution_PYG(in_channels,
                                                                       hidden_channels_list[0],
                                                                       order=order,
                                                                       dropout_p=dropout,
                                                                       non_linearity=non_linearity)])
        for sgcn_in, sgcn_out in zip(hidden_channels_list[:-1], hidden_channels_list[1:]):
            self.conv_list.append(SphericalConvolution_PYG(sgcn_in,
                                                           sgcn_out,
                                                           order=order,
                                                           dropout_p=dropout,
                                                           non_linearity=non_linearity))
        if len(mlp_channels_list) == 1 and mlp_channels_list[0] == 0:
            self.mlp = None
        else:
            self.mlp = MLP(
                channel_list=mlp_channels_list,
                dropout=dropout)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """Apply the network to a graph

        Args:
            x (torch.Tensor): Graph nodes features
            edge_index (torch.Tensor): Graph edge index
            edge_attr (torch.Tensor): Graph edge attributes

        Returns:
            torch.Tensor: The new nodes features
        """
        for conv in self.conv_list:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        if self.mlp is not None:
            x = self.mlp(x)
        return x
