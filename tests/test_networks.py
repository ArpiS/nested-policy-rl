import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import sys
sys.path.append("../simulated_fqi/")
from models.networks import NFQNetwork, ContrastiveNFQNetwork
import matplotlib.pyplot as plt
import numpy as np


def train(x, y, groups, network, optimizer):

    predicted_q_values = network(x, groups).squeeze()
    loss = F.mse_loss(predicted_q_values, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def test_contrastive_network():

    # Setup agent
    network = ContrastiveNFQNetwork(state_dim=0, is_contrastive=True, nonlinearity=nn.Identity)
    optimizer = optim.Rprop(network.parameters())

    # Generate data
    n, m = 100, 100
    beta_shared = -1
    beta_fg = 2.1
    x_bg, x_fg = np.linspace(-3, 3, m), np.linspace(-3, 3, n)
    x = np.concatenate([x_bg, x_fg])
    groups = np.concatenate([np.zeros(m), np.ones(n)])
    y = beta_shared * x + beta_fg * groups * x# + np.random.normal(scale=0.5, size=m+n)

    x = torch.FloatTensor(x).unsqueeze(1)
    y = torch.FloatTensor(y)
    groups = torch.FloatTensor(groups).unsqueeze(1)
    
    for epoch in range(200):

        loss = train(x, y, groups, network, optimizer)
        
        # if epoch % 10 == 0:
        #     print("Epoch: {:4d}, Loss: {:4f}".format(epoch, loss))

    network.eval()
    with torch.no_grad():
        preds = network(x, groups)

    assert np.allclose(preds.squeeze().numpy(), y.squeeze().numpy(), atol=1e-4)
    # plt.scatter(x, preds, c=groups)
    # plt.show()
    # import ipdb; ipdb.set_trace()
    
if __name__ == "__main__":
    test_contrastive_network()
