"""Networks for NFQ."""
import torch
import torch.nn as nn


class NFQNetwork(nn.Module):
    def __init__(self, state_dim):
        """Networks for NFQ."""
        super().__init__()
        self.state_dim = state_dim
        self.layers = nn.Sequential(
            nn.Linear(self.state_dim + 1, 5),
            nn.Sigmoid(),
            nn.Linear(5, 5),
            nn.Sigmoid(),
            nn.Linear(5, 1),
            nn.Sigmoid(),
        )

        # Initialize weights to [-0.5, 0.5]
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.uniform_(m.weight, -0.5, 0.5)

        self.layers.apply(init_weights)

    def forward(self, x: torch.Tensor, group) -> torch.Tensor:
        """
        Forward propagation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of observation and action concatenated.

        Returns
        -------
        y : torch.Tensor
            Forward-propagated observation predicting Q-value.

        """
        return self.layers(x)

class ContrastiveNFQNetwork(nn.Module):
    def __init__(self, state_dim, is_contrastive: bool=True, nonlinearity=nn.Sigmoid):
        super().__init__()
        self.state_dim = state_dim
        LAYER_WIDTH = self.state_dim + 1
        self.is_contrastive = is_contrastive
        self.layers_shared = nn.Sequential(
            nn.Linear(self.state_dim + 1, LAYER_WIDTH),
            nonlinearity()
        )
        if self.is_contrastive:
            self.layers_fg = nn.Sequential(
                nn.Linear(self.state_dim + 1, LAYER_WIDTH),
                nonlinearity()
            )
            self.layers_last = nn.Sequential(
                nn.Linear(LAYER_WIDTH*2, 1),
                nonlinearity()
            )
        else:
            self.layers_last = nn.Sequential(
                nn.Linear(LAYER_WIDTH, 1),
                nonlinearity()
            )

        # Initialize weights to [-0.5, 0.5]
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.uniform_(m.weight, -0.5, 0.5)

        self.layers_shared.apply(init_weights)
        self.layers_last.apply(init_weights)
        if self.is_contrastive:
            self.layers_fg.apply(init_weights)

    def forward(self, x: torch.Tensor, group) -> torch.Tensor:
        x_shared = self.layers_shared(x)
        if self.is_contrastive:
            x_fg = self.layers_fg(x)
            x = torch.cat((x_shared, x_fg * group), dim=-1)
            # x = x_shared + x_fg * group
            
        else:
            x = x_shared

        return self.layers_last(x)




