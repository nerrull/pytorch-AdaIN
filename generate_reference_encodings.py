import time
from os.path import splitext, join, exists
from os import mkdir
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import net



vgg_path = "models/vgg_normalised.pth"
ref_path = "encodings/ref/"

if not exists(ref_path):
    mkdir(ref_path)

torch.cuda.set_device(0)



decoder = net.decoder
vgg = net.vgg

vgg.load_state_dict(torch.load(vgg_path))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(vgg, decoder)
network.cuda()

resolutions =  [64,128,256,512,1024]

for resolution in resolutions:
    input = np.zeros(shape=(3, resolution, resolution), dtype=np.float32)
    input_tensor = torch.from_numpy(input)
    input_tensor = input_tensor.cuda()
    input_variable = Variable(input_tensor.unsqueeze(0), volatile=True)

    for layer_index in range(1,5):
        t = time.time()

        encoding = network.get_layer_encoding(input_variable,layer_index)
        encoding =  encoding.cpu().data.numpy()
        out_file = "reference_encoding_{}_relu{}".format(resolution, layer_index)
        out_path = join(ref_path, out_file)
        np.save(out_path, encoding)
        t = time.time()-t
        print("Took {} seconds to extract encoding {}, relu_{}".format(t, resolution, layer_index))



