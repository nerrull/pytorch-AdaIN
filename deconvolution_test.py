import time
from os import mkdir
from os.path import basename, splitext, join, exists
import random
import numpy as np
import torch
from torch.autograd import Variable
import deconvolutional_net as net
import cv2 as cv
import torch.nn as nn
from torchvision.utils import save_image
from utils import test_transform
from PIL import Image


resolution = 512
amplitude =10.
layer = 4

decoder_path = "models/vgg_deconv.pth"
vgg_path = "models/vgg_normalised.pth"
ref_path = "encodings/ref/"
content_dir = "input/content/"

encoding_dir = "encodings"
output_dir = join("output", "channel_activations_noise_norm")
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

image_name = "avril.jpg"
image_path  = join(content_dir, image_name)
img = cv.imread(image_path)
channels = img.shape[2]
if channels == 1:
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
elif channels == 4:
    img = cv.cvtColor(img, cv.COLOR_BGRA2RGB)
else:
    print("gm")
    #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)



image_tf = test_transform((resolution,resolution), False)


input_tensor = image_tf(Image.fromarray(img))
input_tensor = input_tensor.cuda()
input_variable = Variable(input_tensor.unsqueeze(0), volatile=True)



# cv.putText(frame, "{}".format(neuron_index), (15,15), cv.FONT_HERSHEY_PLAIN, 1.,(1.,1.,1.))
cv.imshow("Input", img)
cv.waitKey(0)


reference_input = join(ref_path, "reference_encoding_{}_relu{}.npy".format(resolution, layer))
reference_input = np.load(reference_input)
num_neurons = reference_input.shape[1]



for activated_neuron_index in range(num_neurons):

    neuron_index = activated_neuron_index

    t = time.time()


    image = network.mini_pass(input_variable)
    #image = network.test_pass(input_variable)
    frame = image.cpu().data.numpy()[0]
    minimum =  np.min(frame)
    maximum = np.max(frame)
    frame = (frame -minimum)/(maximum - minimum)

    frame = np.transpose(frame,  (1,2,0))


    t = time.time() - t
    print("Took {} seconds to extract encoding".format(t))
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    cv.putText(frame, "{}".format(neuron_index), (15,15), cv.FONT_HERSHEY_PLAIN, 1.,(1.,1.,1.))
    cv.imshow("Neuron activation", frame)
    cv.waitKey(33)
    out_file = "layer4_channel{}.png".format(str(activated_neuron_index).zfill(3))
    out_path = join(output_dir, out_file)
    #save_image(image,out_path, normalize=True)
    #cv.imwrite(out_path, frame)
    del image, frame
    torch.cuda.empty_cache()
