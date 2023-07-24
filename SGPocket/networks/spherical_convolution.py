import torch
import torch.nn as nn
import torch_geometric as pyg

# Based on https://gitlab.inria.fr/GruLab/s-gcn/-/blob/master/src/sgcn/layers.py


class SphericalConvolution_PYG(pyg.nn.MessagePassing):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 order: int,
                 dropout_p: float,
                 non_linearity: str,
                 bias: bool = True):
        """Constructor

        Args:
            in_features (int): Input features size
            out_features (int): Output features size
            order (int): Spherical harmonics order
            dropout_p (float): Dropout
            non_linearity (str): Non linearity
            bias (bool, optional): Bias. Defaults to True.
        """
        assert non_linearity in ['relu', 'elu', 'sigmoid',
                                 'tanh', 'mish', 'none'], 'Incorrect non-linearity'

        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.in_features = in_features
        self.out_features = out_features
        self.order_squared = order ** 2

        self.non_linearity = None
        self.non_linearity_name = non_linearity
        if non_linearity == 'relu':
            self.non_linearity = nn.functional.relu
        elif non_linearity == 'elu':
            self.non_linearity = nn.functional.elu
        elif non_linearity == 'sigmoid':
            self.non_linearity = torch.sigmoid
        elif non_linearity == 'tahn':
            self.non_linearity = torch.tanh
        elif non_linearity == 'none':
            self.non_linearity = None
        self.lin_part = torch.nn.Linear(
            in_features, out_features, bias=False)  # = H^{k-1}W
        self.lins = torch.nn.ModuleList()

        for _ in range(self.order_squared):
            self.lins.append(torch.nn.Linear(
                in_features, out_features, bias=False))  # W_l^m

        self.dropout = torch.nn.Dropout(dropout_p)

        self.bias = nn.Parameter(torch.Tensor(
            out_features)) if bias else None  # b

        torch.nn.init.xavier_normal_(self.lin_part.weight)
        for i in range(self.order_squared):
            torch.nn.init.xavier_normal_(self.lins[i].weight)
        if bias:
            torch.nn.init.constant_(self.bias, 0)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """Apply the layer to a graph

        Args:
            x (torch.Tensor): Graph nodes features
            edge_index (torch.Tensor): Graph edge index
            edge_attr (torch.Tensor): Graph edge attributes

        Returns:
            torch.Tensor: The new nodes features
        """
        x_part = self.lin_part(x)
        out = self.propagate(edge_index=edge_index, x=x,
                             edge_attr=edge_attr, aggr='add')
        out += x_part
        if self.bias is not None:
            out = out + self.bias
        if self.non_linearity is not None:
            out = self.non_linearity(out)
        out = self.dropout(out)
        return out

    def message(self,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """Create a message from neighbors

        Args:
            x_j (torch.Tensor): Neighbor features
            edge_attr (torch.Tensor): Edge attributes

        Returns:
            torch.Tensor: The message
        """
        edge_ponderation = []
        for i in range(self.order_squared):
            edge_ponderation.append(edge_attr[:, i].view(-1, 1) * x_j)
        return torch.stack(edge_ponderation)

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Update the node features

        Args:
            aggr_out (torch.Tensor): Neighbors message

        Returns:
            torch.Tensor: New node features
        """
        conv_sh = []
        for i in range(self.order_squared):
            conv_sh.append(self.lins[i](aggr_out[i]))
        return torch.stack(conv_sh).sum(axis=0)

    def __repr__(self) -> str:
        """Returns a str representing the layer

        Returns:
            str: A str representing the layer
        """
        return "Weight:\t{}\nWeights:\t{}".format(self.lin_part, self.lins)
