import time
import os
from os.path import basename
from os.path import splitext, join, exists

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from utils import test_transform
import net
from function import adaptive_instance_normalization, adaptive_instance_content_normalization
from function import coral

import cv2 as cv



vgg_path = "models/vgg_normalised.pth"
content_size = 256
crop = True
content_dir = "input/content/"

torch.cuda.set_device(0)

vgg = net.vgg
vgg.eval()
vgg.load_state_dict(torch.load(vgg_path))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.cuda()


for file in os.listdir(content_dir):
    t = time.time()

    image_name = file.split(".")[0:-1][0]
    out_file = "encodings/{}_{}".format(image_name,content_size)
    if exists(out_file +".npy"):
        print("{} already calculated, skipping".format(out_file))
        continue

    image_tf = test_transform(content_size, crop)

    img = cv.imread(join(content_dir, file))
    channels =img.shape[2]
    if channels ==1:
        image = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    elif channels ==4:
        image = cv.cvtColor(img, cv.COLOR_BGRA2RGB)
    else:
        image = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    image = image_tf(Image.fromarray(image))

    image = image.cuda()
    image = Variable(image.unsqueeze(0), volatile=True)
    image_f =  vgg(image)
    #image_f_norm = adaptive_instance_content_normalization(image_f)

    features =  image_f.cpu().data.numpy()

    np.save(out_file, features)

    t = time.time()-t
    print("Took {} seconds to extract encoding".format(t))
