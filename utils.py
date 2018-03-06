from torchvision import transforms
import numpy as np

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def generate_2d_sin(dims, period, mult =1.0):
    x = np.arange(0,dims)*2*np.pi*period/dims
    sin1 = np.sin(x)
    sin2 = np.sin(x*mult)
    out = np.zeros((dims,dims))
    for i in range(dims):
        for j in range(dims):
            out[i,j] = ((sin1[i]*sin2[j]) +1 )/2
    return out