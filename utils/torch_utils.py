import torch
from torch import nn

PI = torch.acos(torch.zeros(1)).item() * 2


class SphericPad(nn.Module):
    """Pads spherically the input on all sides with the given padding size."""

    def __init__(self, padding_size):
        super(SphericPad, self).__init__()
        if isinstance(padding_size, int) or (isinstance(padding_size, torch.Tensor) and padding_size.shape==()):
            self.pad_left = self.pad_right = self.pad_top = self.pad_bottom = self.pad_front = self.pad_back = padding_size
        elif (isinstance(padding_size, tuple) or isinstance(padding_size, torch.Tensor)) and len(padding_size) == 3:
            self.pad_left = self.pad_right = padding_size[0]
            self.pad_top = self.pad_bottom = padding_size[1]
            self.pad_front = self.pad_back = padding_size[2]
        elif (isinstance(padding_size, tuple) or isinstance(padding_size, torch.Tensor)) and len(padding_size) == 6:
            self.pad_left = padding_size[0]
            self.pad_top = padding_size[1]
            self.pad_right = padding_size[2]
            self.pad_bottom = padding_size[3]
            self.pad_front = padding_size[4]
            self.pad_back = padding_size[5]
        else:
            raise ValueError('The padding size shoud be: int, torch.IntTensor  or tuple of size 2 or tuple of size 4')

    def forward(self, input):

        output = torch.cat([input, input[:, :self.pad_bottom, :, :]], dim=1)
        output = torch.cat([output, output[:, :, :self.pad_right, :]], dim=2)
        output = torch.cat([output, output[:, :, :, :self.pad_back]], dim=3)
        output = torch.cat([output[:, -(self.pad_bottom + self.pad_top):-self.pad_bottom, :, :], output], dim=1)
        output = torch.cat([output[:, :, -(self.pad_right + self.pad_left):-self.pad_right, :], output], dim=2)
        output = torch.cat([output[:, :, :, -(self.pad_back + self.pad_front):-self.pad_right], output], dim=3)

        return output