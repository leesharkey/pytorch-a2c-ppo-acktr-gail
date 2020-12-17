import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
import math


class CustomGRU(nn.Module):

    def __init__(self, recurrent_input_size, hidden_size):
        super(CustomGRU, self).__init__()
        # Hidden dimensions
        self.hidden_size = hidden_size

        self.gru_cell = GRUCell(recurrent_input_size, hidden_size)

    def forward(self, x, h0):
        """Steps the GRUCell forward n times according to number of obs.

        Args:
            x: Observations. Shape either [numproc, obssize] or
                [num_timesteps, numproc, obssize]
            h0: Initial state.

        Returns:
            True if successful, False otherwise.

        """
        outs = []
        ht = h0

        if len(x.shape) == 2:
            tsteps = 1
        else:
            tsteps = x.size(0)


        for t in range(tsteps):
            ht, rt, zt = self.gru_cell(x[t, :, :], ht)
            outs.append(ht)

        out = outs[-1].squeeze()

        return out


class GRUCell(nn.Module):
    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        """Steps the GRUCell forward 1 time.

        Args:
            x: Observations. Shape either [numproc, obssize] or
                [num_timesteps, numproc, obssize]
            h: Current hidden state.

        Returns:
            True if successful, False otherwise.

        """
        #TODO implement GRU with right dimensions etc. and returning the
        # desired properties.
        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy

