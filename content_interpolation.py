import time
from os import mkdir
from os.path import basename, splitext, join, exists


import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from utils import test_transform
import net
from function import adaptive_instance_normalization
from function import coral
import cv2 as cv

class interpolate_linear:
    def __init__(self, start_encoding, end_encoding, num_steps):
        self.start =start_encoding
        self.step = (end_encoding - start_encoding) / num_steps

    def __iter__(self):
        return self.next()

    def __next__(self):
        self.next()
    def next(self):
        for i in range(num_steps):
            yield self.start + i*self.step

class interpolate_channels:
    def __init__(self, start_encoding, end_encoding, num_steps):
        self.start =start_encoding
        self.end = end_encoding
        self.step = (end_encoding - start_encoding) / num_steps
        self.channel_count = start_encoding.shape[1]
        self.channel_step = self.channel_count/num_steps

    def __iter__(self):
        return self.next()

    def __next__(self):
        self.next()

    def next(self):
        encoding = self.start
        for i in range(num_steps):
            encoding[0,:i*self.channel_step,:,:]= end_encoding[0,:i*self.channel_step,:,:]
            yield encoding

class mask:
    def __init__(self, start_encoding, end_encoding, num_steps):
        self.start =start_encoding
        self.end = end_encoding
        self.step = (end_encoding - start_encoding) / num_steps
        self.channel_count = start_encoding.shape[1]
        self.channel_step = self.channel_count/num_steps

    def __iter__(self):
        return self.next()

    def __next__(self):
        self.next()

    def next(self):
        for i in range(self.channel_count):
            encoding = np.zeros_like(self.start)
            print("Min: {}, Max: {}".format(np.min(start_encoding[0,i,:,:]), np.max(start_encoding[0,i,:,:])))
            encoding[0,i,:,:]=start_encoding[0,i,:,:]
            yield encoding

decoder_path = "models/decoder.pth"
content_size = 256
crop = False

encoding_dir = "encodings"
output_dir = join(encoding_dir, "out")
if not exists(output_dir):
    mkdir(output_dir)

start_enc_path = join(encoding_dir, "bouba_green_{}.npy".format(content_size))
end_enc_path = join(encoding_dir, "kiki_blue_{}.npy".format(content_size))

start_encoding = np.load(start_enc_path)
end_encoding = np.load(end_enc_path)


start_encoding_variable = Variable(torch.from_numpy(start_encoding), volatile=True).cuda()
end_encoding_variable = Variable(torch.from_numpy(end_encoding), volatile=True).cuda()

torch.cuda.set_device(0)
decoder = net.decoder
decoder.eval()
decoder.load_state_dict(torch.load(decoder_path))
decoder.cuda()

#
# cap = cv.VideoCapture(0)
# writer = cv.VideoWriter(join(output_dir,'output.avi'), -1, 20.0, (content_size, content_size))

num_steps = 50
#interpolator_lin = interpolate_linear(start_encoding, end_encoding, num_steps)

interpolator_chan = mask(start_encoding, end_encoding, num_steps)
# interps_zipped= zip(interpolator_lin, interpolator_chan)

for i, encoding_1 in enumerate(interpolator_chan):

    t = time.time()

    enc1 = Variable(torch.from_numpy(encoding_1), volatile=True).cuda()
    # enc2= Variable(torch.from_numpy(encoding_2), volatile=True).cuda()

    output = decoder(enc1).data
    output = output.cpu()
    frame = np.transpose(output.numpy()[0], (1,2,0))
    cv.imshow("current step linear", cv.cvtColor( frame, cv.COLOR_RGB2BGR))

    # output = decoder(enc2).data
    # output = output.cpu()
    # frame = np.transpose(output.numpy()[0], (1,2,0))
    # cv.imshow("current step_chan", cv.cvtColor( frame, cv.COLOR_RGB2BGR))

    t = time.time()-t
    print("Took {} seconds to extract encoding".format(t))


    cv.waitKey(10)
    # out_file = join(output_dir, "out_{}.png".format(str(i).zfill(3)))
    # save_image(output, out_file)