import numpy as np
from scipy import signal, ndimage
from skimage import restoration
import matplotlib.pyplot as plt
image_path = "input/content/avril.jpg"
image = ndimage.imread(image_path)

#do the test on jsut one channel
image = image[:,:,0]/255


kernel = np.random.rand(3,3)-0.5


kernel2 = np.array([[1,1,1],
                    [0,1,1],
                    [0,0,1]])

kernel2_spectrum = np.fft.fft2(kernel2)

kernel2_inverse_spectrum = 1/kernel2_spectrum

kernel2_inverse = np.real(np.fft.ifft2(kernel2_inverse_spectrum))

inv = np.linalg.inv(kernel2)
prod = np.matmul(kernel2,inv)

image_padded = np.pad(image,1, 'reflect')
image_conv = signal.convolve2d(image_padded, kernel2, 'valid')

# max = np.max(image_conv)
# min = np.min(image_conv)
# image_conv = (image_conv -min)/(max-min)


image_conv_padded = np.pad(image_conv,1, 'reflect')

image_deconv = signal.convolve2d(image_conv_padded, kernel2_inverse, 'valid')
# max = np.max(image_deconv)
# min = np.min(image_deconv)
# image_deconv = (image_deconv -min)/(max-min)

image_deconv_spectrum = np.fft.fft2(image_conv)/kernel2_spectrum
image_deconv = np.real(np.fft.ifft2(image_deconv_spectrum))

diff = image - image_deconv

plt.subplot(221)
im =plt.imshow(image)
plt.title("original")
plt.colorbar(im)

plt.subplot(222)
im = plt.imshow(image_conv)
plt.title("conv")
plt.colorbar(im)

plt.subplot(223)
im=plt.imshow(image_deconv)
plt.title("deconv")
plt.colorbar(im)

plt.subplot(224)
im= plt.imshow(diff)
plt.colorbar(im)
plt.title("diff")
plt.show()