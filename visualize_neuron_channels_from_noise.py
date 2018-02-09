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
amplitude =30.
layer = 4

decoder_path = "models/decoder.pth"
vgg_path = "models/vgg_normalised.pth"
ref_path = "encodings/ref/"

encoding_dir = "encodings"
output_dir = join("output", "layer{}_channel_activations_noise_norm_30".format(layer))
if not exists(output_dir):
    mkdir(output_dir)

decoder = net.decoder
vgg = net.vgg

vgg.load_state_dict(torch.load(vgg_path))
vgg = nn.Sequential(*list(vgg.children())[:31])
decoder.load_state_dict(torch.load(decoder_path))
network = net.Net(vgg, decoder)
network.cuda()



input = np.random.rand(3, resolution, resolution).astype(np.float32)
input_tensor = torch.from_numpy(input)
input_tensor = input_tensor.cuda()
input_variable = Variable(input_tensor.unsqueeze(0), volatile=True)

frame = np.transpose(input,  (1,2,0))
frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
# cv.putText(frame, "{}".format(neuron_index), (15,15), cv.FONT_HERSHEY_PLAIN, 1.,(1.,1.,1.))
cv.imshow("Neuron activation", frame)
cv.waitKey(0)


reference_input = join(ref_path, "reference_encoding_{}_relu{}.npy".format(resolution, layer))
reference_input = np.load(reference_input)
num_neurons = reference_input.shape[1]


for activated_neuron_index in range(num_neurons):

    neuron_index = activated_neuron_index

    # input = np.random.rand(3, resolution, resolution).astype(np.float32)
    # input_tensor = torch.from_numpy(input)
    # input_tensor = input_tensor.cuda()
    # input_variable = Variable(input_tensor.unsqueeze(0), volatile=True)

    t = time.time()

    mask_array = np.ones_like(reference_input).astype(np.uint8)
    mask_array[:,neuron_index,:,:] = 0
    mask_tensor = torch.from_numpy(mask_array).cuda()

    image = network.get_channel_activation(input_variable, mask_tensor,layer, amplitude)
    frame = image.cpu().data.numpy()[0]
    frame = np.transpose(frame,  (1,2,0))


    t = time.time() - t
    print("Took {} seconds to extract encoding".format(t))
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    cv.putText(frame, "{}".format(neuron_index), (15,15), cv.FONT_HERSHEY_PLAIN, 1.,(1.,1.,1.))
    cv.imshow("Neuron activation", frame)
    cv.waitKey(33)
    out_file = "layer{}_channel{}.png".format(layer, str(activated_neuron_index).zfill(3))
    out_path = join(output_dir, out_file)
    save_image(image.data,out_path, normalize=True)
    #cv.imwrite(out_path, frame)
    del image, mask_tensor, frame
    torch.cuda.empty_cache()
