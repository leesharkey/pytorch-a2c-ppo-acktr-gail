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
        hxs = []
        rs = []
        zs = []
        hhats = []
        ht = h0

        for t in range(x.size(0)):
            ht, rt, zt, hhat = self.gru_cell(x[t, :, :], ht)
            hxs.append(ht)
            rs.append(rt)
            zs.append(zt)
            hhats.append(hhat)
        last_hx = hxs[-1]
        hxs = torch.cat(hxs)
        internals = {'r': torch.cat(rs),
                     'z': torch.cat(zs),
                     'hhat': torch.cat(hhats)}
        return hxs, last_hx, internals


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

        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        if gate_x.size(0) != 1:  # These conditionals are for num-processes=1
            gate_x = gate_x.squeeze()
        if len(gate_h.shape)==3:
            gate_h = gate_h.squeeze(0)

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        rt = F.sigmoid(i_r + h_r)  # reset gate vector
        zt = F.sigmoid(i_i + h_i)  # update gate vector
        hhat = F.tanh(i_n + (rt * h_n))  # candidate activation vector

        ht = hhat + zt * (hidden - hhat)

        return ht, rt, zt, hhat

