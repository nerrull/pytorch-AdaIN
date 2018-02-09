import time
from os import mkdir
from os.path import basename, splitext, join, exists
import random
import numpy as np
import torch
from torch.autograd import Variable
import net
import cv2 as cv
from torchvision.utils import save_image


class mask:
    def __init__(self, ref_encoding, ):
        self.ref = ref_encoding
        self.channel_count = ref_encoding.shape[1]
        self.shape = ref_encoding[0, 0, :, :].shape

        sixteenth_seed = np.random.rand(int(self.shape[0]/4), int(self.shape[1]/4))*25 +25
        eighth_seed = np.concatenate((sixteenth_seed,sixteenth_seed),axis=0)
        quarter_seed = np.concatenate((eighth_seed,eighth_seed),axis=1)
        half_seed = np.concatenate((quarter_seed,quarter_seed),axis=0)
        self.seed =  np.concatenate((half_seed,half_seed),axis=1)


    def __iter__(self):
        return self.next()

    def __next__(self):
        self.next()

    def next(self):
        for i in range(self.channel_count):
            encoding = np.zeros_like(self.ref)
            seed = self.seed

            sixteenth_seed = np.random.rand(int(self.shape[0] / 4), int(self.shape[1] / 4)) * 25 + 25
            eighth_seed = np.concatenate((sixteenth_seed, sixteenth_seed), axis=0)
            quarter_seed = np.concatenate((eighth_seed, eighth_seed), axis=1)
            half_seed = np.concatenate((quarter_seed, quarter_seed), axis=0)
            seed = np.concatenate((half_seed, half_seed), axis=1)
            encoding[0,0,:,:]= seed
            #encoding[0,random.randint(0,self.channel_count-1),:,:]= seed
            yield encoding



decoder_path = "models/decoder.pth"
content_size = 256
crop = False

encoding_dir = "encodings"
output_dir = join("output", "channel_activations")
if not exists(output_dir):
    mkdir(output_dir)

start_enc_path = join(encoding_dir, "avril_{}.npy".format(content_size))

ref_encoding = np.load(start_enc_path)



torch.cuda.set_device(0)
decoder = net.decoder
decoder.eval()
decoder.load_state_dict(torch.load(decoder_path))
decoder.cuda()


mask_channel = mask(ref_encoding)

for i, encoding in enumerate(mask_channel):

    t = time.time()

    enc1 = Variable(torch.from_numpy(encoding), volatile=True).cuda()

    output = decoder(enc1).data
    output = output.cpu()
    frame = np.transpose(output.numpy()[0], (1,2,0))
    cv.imshow("current step linear", cv.cvtColor( frame, cv.COLOR_RGB2BGR))
    t = time.time()-t
    print("Took {} seconds to extract encoding".format(t))


    cv.waitKey(100)
    out_file = join(output_dir, "channel_{}.png".format(str(i).zfill(3)))
    #save_image(output, out_file)