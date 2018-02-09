import deconvolutional_net as net
import torch
import torch.nn as nn
from torch import from_numpy
import torch.nn.functional as F

import numpy as np
from deconvolutional_net import AddBias
vgg_path = "models/vgg_normalised.pth"

vgg = net.vgg
vgg.eval()
vgg.load_state_dict(torch.load(vgg_path))

deconv = []
vgg_layers = list(vgg.children())[2:30]
for layer in vgg_layers:
    layer_name = "{}".format( layer.__repr__())

    if "Conv" in layer_name:
        newLayer = nn.Conv2d( layer.out_channels, layer.in_channels, layer.kernel_size)
        weights = layer.weight.data.numpy()
        weights_transposed = from_numpy(np.transpose(weights, (1,0,2,3)))
        newLayer.weight.data= weights_transposed
        newLayer.bias.data= from_numpy(np.zeros_like(newLayer.bias.data.numpy()))


        inverse_bias =AddBias(in_features=layer.out_channels)
        inverse_bias.bias.data = layer.bias.data*-1
        deconv.append(newLayer)
        if layer.kernel_size[0]>1:
            deconv.append(nn.ReflectionPad2d((1,1,1,1)))
        deconv.append(inverse_bias)


    elif "ReLU" in layer_name:
        newLayer = nn.ReLU()
        deconv.append(newLayer)

    elif "MaxPool" in layer_name:

        newLayer = nn.MaxUnpool2d(layer.kernel_size, layer.stride,layer.padding)
        deconv.append(newLayer)
    elif "ReflectionPad" in layer_name:

        #newLayer = nn.ReflectionPad2d(layer.padding)
        #deconv.append(newLayer)
        continue
    else:
        print("unhandled layer")
    #print("{}->{}".format(layer, newLayer))


deconv.reverse()
for layer in deconv: print(layer)
n=nn.Sequential(*deconv)


torch.save(n.state_dict(), "models/vgg_deconv"+ '.pth')
