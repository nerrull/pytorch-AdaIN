import time
from os import mkdir
from os.path import basename, splitext, join, exists
import random
import numpy as np
import torch
from torch.autograd import Variable
import net
import cv2 as cv
import torch.nn as nn
from torchvision.utils import save_image



resolution = 256
decoder_path = "models/decoder.pth"
vgg_path = "models/vgg_normalised.pth"
ref_path = "encodings/ref/"

encoding_dir = "encodings"
output_dir = join("output", "channel_activations")
if not exists(output_dir):
    mkdir(output_dir)

torch.cuda.set_device(0)
decoder = net.decoder
vgg = net.vgg

vgg.load_state_dict(torch.load(vgg_path))
vgg = nn.Sequential(*list(vgg.children())[:31])
decoder.load_state_dict(torch.load(decoder_path))
network = net.Net(vgg, decoder)
network.cuda()

layer = 3
activated_neuron_index = 5
reference_input = join(ref_path, "reference_encoding_{}_relu{}.npy".format(resolution, layer))
reference_input = np.load(reference_input)
channel_shape = reference_input[0,0, :,:].shape

num_neurons = reference_input.shape[1]

amplitude =100.
neuron_1 = 13
neuron_2 = 54
simulated_neuron_output = np.random.rand(channel_shape[0], channel_shape[1]) * amplitude

numsteps = 500.

for i in range(int(numsteps)):
    t = time.time()


    input = np.zeros_like(reference_input)
    input[:, neuron_1, :, :] = simulated_neuron_output * i/numsteps
    input[:, neuron_2, :, :] = simulated_neuron_output * (numsteps-i)/numsteps


    input_tensor = torch.from_numpy(input)
    input_tensor = input_tensor.cuda()
    input_variable = Variable(input_tensor, volatile=True)

    image = network.get_image_from_encoder_layer(input_variable, layer)

    frame = np.transpose(image,  (1,2,0))
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    cv.putText(frame, "{}".format(i), (15, 15), cv.FONT_HERSHEY_PLAIN, 1., (1., 1., 1.))
    cv.imshow("current step linear", cv.cvtColor(frame, cv.COLOR_RGB2BGR))
    t = time.time() - t
    print("Took {} seconds to extract encoding".format(t))

    cv.waitKey(33)
