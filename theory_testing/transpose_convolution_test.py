import numpy as np
from numpy import arange
from math import pi
from matplotlib import pyplot as plt
from torch import from_numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



def pad1d(tensor, pad, permute_dims=True):
    # tensor should be in shape (batch, time, feat)
    # pad should be in shape (left, right)
    if permute_dims:
        tensor = tensor.permute(0, 2, 1).contiguous() # get features on first dim since we are padding time
    else:
        tenosr = tensor.contiguous()
    original_size = tensor.size() # (batch, feat, time)
    final_new_size = (original_size[0], original_size[1], original_size[2] + pad[0] + pad[1])
    temp_new_size = original_size[:2] + (1,) + original_size[2:]
    assert len(temp_new_size) == 4
    tensor = tensor.view(*temp_new_size)
    pad = pad + (0, 0)
    tensor = F.pad(tensor, pad)
    tensor = tensor.view(*final_new_size)
    if permute_dims:
        tensor = tensor.permute(0, 2, 1)
    return tensor

array = [arange(1,1000).astype(np.float32)*2*pi/1000.]

sin_wave = np.sin(array)

input = Variable(from_numpy(sin_wave).unsqueeze(0), volatile =True)

channels = 1
layer = nn.Conv1d(1, channels,20)
#layer.weight.data = from_numpy(weights)
weight = layer.weight.data.numpy()

weight_t = np.transpose(layer.weight.data.numpy(), (0,1,2))

transpose_layer =  nn.ConvTranspose1d(channels, 1,20)
transpose_layer.weight.data = from_numpy(weight_t)
transpose_layer.bias.data = from_numpy(layer.bias.data.numpy())

#pad1d(input,(1,1), mode="reflect")
conv = layer(input)
deconv = transpose_layer(conv)


plt.subplot(211)
plt.plot(sin_wave[0])

plt.subplot(212)

plt.plot(deconv.data.numpy()[0][0])

plt.show()

