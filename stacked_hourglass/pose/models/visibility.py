import torch.nn as nn
import torch.nn.functional as F





class JointVisibilityNet(nn.Module):
    """
    A neural network that will take the set of probability maps output by the stacked hourglass network, and predict
    a joint visibility from this.

    Note that the network runs PER joint. So a batch size of N for the 2D predictions will lead to a batch size of
    N*num_joints for this network.

    This is a simple 3 layer network with fixed hidden size.
    """
    def __init__(self, hourglass_stacks, hourglass_prob_distr_shape=[64,64], hidden_size=1024):
        super(JointVisibilityNet, self).__init__()
        self.input_dim = hourglass_stacks * hourglass_prob_distr_shape[0] * hourglass_prob_distr_shape[1]
        self.hidden_size = hidden_size

        self.relu = nn.ReLU(inplace=True)
        self.W1 = nn.Linear(self.input_dim, self.hidden_size)
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.bn2 = nn.BatchNorm1d(self.hidden_size)
        self.W3 = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        """
        Assume an input of shape (Nj, stacks, height, width), where Nj is the total batch size (over actual different
        poses and different joints in each pose). We reshape to (Nj, input_dim) at the beginning
        """
        x = x.view((-1, self.input_dim))
        x = self.W1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.W2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.W3(x)
        return x # F.sigmoid(x) # return the logits, rather than the sigmoid
