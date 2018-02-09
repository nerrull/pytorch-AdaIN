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
from noise import perlin
import matplotlib.pyplot as plt


resolution = 256
decoder_path = "models/decoder.pth"
vgg_path = "models/vgg_normalised.pth"
ref_path = "encodings/ref/"

encoding_dir = "encodings"
output_dir = join("./output", "video")
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
activated_neuron_index = 13
reference_input = join(ref_path, "reference_encoding_{}_relu{}.npy".format(resolution, layer))
reference_input = np.load(reference_input)
channel_shape = reference_input[0,0, :,:].shape
width = channel_shape[0]
height = channel_shape[1]

num_neurons = reference_input.shape[1]

amplitude =100.
neuron = 13
numsteps = 3600.

simulated_neuron_output = np.random.randn(channel_shape[0], channel_shape[1])

lin = np.linspace(0,5,width,endpoint=False).astype(np.float32)
x,y = np.meshgrid(lin,lin)
seed = 0
#simulated_neuron_output = perlin(x,y,seed)

mask = np.zeros_like(simulated_neuron_output)
for r in range(width):
    for c in range(height):
        mask[r,c]= abs(width/2 -r) + abs(height/2-c)

mask = 1- mask/np.max(mask).astype(np.float32)

# cv.imshow("mask", mask)
# cv.waitKey(0)

masked_in = mask * simulated_neuron_output

input = np.zeros_like(reference_input)
input[:, neuron, :, :] = simulated_neuron_output

for i in range(int(numsteps)+1):
    t = time.time()
    seed +=1


    # increment = np.random.randn(channel_shape[0], channel_shape[1])*0.03
    increment =perlin(x, y, seed).astype(np.float32)*0.1
    input[:, neuron, :, :] = input[:, neuron, :, :] + increment
    input =np.clip(input, 0.0,1.0)


    input_tensor = torch.from_numpy(input)
    input_tensor = input_tensor.cuda()
    input_variable = Variable(input_tensor, volatile=True)

    image = network.get_image_from_encoder_layer(input_variable, layer, amplitude)
    frame = image.cpu().data.numpy()[0]
    frame = np.transpose(frame,  (1,2,0))
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    #cv.putText(frame, "{}".format(i), (15, 15), cv.FONT_HERSHEY_PLAIN, 1., (1., 1., 1.))
    cv.imshow("current step linear",frame)
    t = time.time() - t
    print("Took {} seconds to extract encoding".format(t))

    cv.waitKey(33)

    out_file = join(output_dir, "brain_{}.png".format(str(i).zfill(4)))
    #save_image(image.data, out_file)
