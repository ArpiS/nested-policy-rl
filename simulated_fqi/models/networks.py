"""Networks for NFQ."""
import torch
import torch.nn as nn
import numpy as np


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
        self.freeze_shared = False
        self.freeze_fg = False

        self.layers_shared = nn.Sequential(
                nn.Linear(self.state_dim + 1, LAYER_WIDTH),
                nonlinearity(),
                nn.Linear(LAYER_WIDTH, LAYER_WIDTH),
                nonlinearity()
            )
        self.layers_fg = nn.Sequential(
            nn.Linear(self.state_dim+1, LAYER_WIDTH),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH, LAYER_WIDTH),
            nonlinearity()
        )
        self.layers_last_shared = nn.Sequential(
            nn.Linear(LAYER_WIDTH, 1),
            nonlinearity()
        )
        self.layers_last_fg = nn.Sequential(
            nn.Linear(LAYER_WIDTH, 1),
            nonlinearity()
        )
        self.layers_last = nn.Sequential(
            nn.Linear(LAYER_WIDTH*2, 1),
            nonlinearity()
        )
        # Initialize weights to [-0.5, 0.5]
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.uniform_(m.weight, -0.5, 0.5)
        
        def init_weights_fg(m):
            if type(m) == nn.Linear:
                torch.nn.init.zeros_(m.weight)

        self.layers_shared.apply(init_weights)
        
        
        #if self.is_contrastive:
        self.layers_last_shared.apply(init_weights)
        self.layers_fg.apply(init_weights_fg)
        self.layers_last_fg.apply(init_weights_fg)
        self.layers_last.apply(init_weights)

        if is_contrastive:
            for param in self.layers_fg.parameters():
                param.requires_grad = False
            for param in self.layers_last_fg.parameters():
                param.requires_grad = False
        #else:
        #    self.layers_last.apply(init_weights)


        

    def forward(self, x: torch.Tensor, group=0) -> torch.Tensor:

        x_shared = self.layers_shared(x)

        if not self.is_contrastive:
            group = 1

        x_shared = self.layers_last_shared(x_shared)

        x_fg = self.layers_fg(x)
        x_fg = self.layers_last_fg(x_fg)
        return x_shared + x_fg * group
        # x = torch.cat((x_shared, x_fg * group), dim=-1)
        # return self.layers_last(x)
        

    def freeze_shared_layers(self):
        for param in self.layers_shared.parameters():
            param.requires_grad = False
        for param in self.layers_last_shared.parameters():
            param.requires_grad = False

    def unfreeze_fg_layers(self):
        for param in self.layers_fg.parameters():
            param.requires_grad = True
        for param in self.layers_last_fg.parameters():
            param.requires_grad = True
    
    def freeze_last_layers(self):
        for param in self.layers_last_shared.parameters():
            param.requires_grad = False
        for param in self.layers_last_fg.parameters():
            param.requires_grad = False
    
    def unfreeze_last_layers(self):
        for param in self.layers_last_shared.parameters():
            param.requires_grad = True
        for param in self.layers_last_fg.parameters():
            param.requires_grad = True

    def assert_correct_layers_frozen(self):
            
        if not self.is_contrastive:
            for param in self.layers_fg.parameters():
                assert param.requires_grad == True
            for param in self.layers_last_fg.parameters():
                assert param.requires_grad == True
            for param in self.layers_shared.parameters():
                assert param.requires_grad == True
            for param in self.layers_last_shared.parameters():
                assert param.requires_grad == True

        elif self.freeze_shared:
            for param in self.layers_fg.parameters():
                assert param.requires_grad == True
            for param in self.layers_last_fg.parameters():
                assert param.requires_grad == True
            for param in self.layers_shared.parameters():
                assert param.requires_grad == False
            for param in self.layers_last_shared.parameters():
                assert param.requires_grad == False
        else:

            for param in self.layers_fg.parameters():
                assert param.requires_grad == False
            for param in self.layers_last_fg.parameters():
                assert param.requires_grad == False
            for param in self.layers_shared.parameters():
                assert param.requires_grad == True
            for param in self.layers_last_shared.parameters():
                assert param.requires_grad == True




