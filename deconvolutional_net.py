import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from function import adaptive_instance_normalization as adain
from function import calc_mean_std
import numpy as np
import torch
from torch import from_numpy




class AddBias(torch.nn.Module):
    def __init__(self, in_features):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(AddBias, self).__init__()
        self.in_features =in_features
        self.bias = Parameter(torch.Tensor(in_features))

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        x_shape = x.data.shape
        bias_array= np.zeros(x_shape, dtype=np.float32)
        for i in range(self.bias.data.shape[0]):
            bias_array[:,i,:,:]= self.bias.data[i]
        bias_tensor = from_numpy(bias_array).cuda()
        x.data +=bias_tensor
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features)+ ')'



decoder = nn.Sequential(
    AddBias(in_features=512),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.MaxUnpool2d((2, 2), (2, 2),(0, 0)),
    AddBias(in_features=256),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    AddBias(in_features=256),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    AddBias(in_features=256),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    AddBias(in_features=256),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.MaxUnpool2d((2, 2), (2, 2),(0, 0)),
    AddBias(in_features=128),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    AddBias(in_features=128),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.MaxUnpool2d((2, 2), (2, 2),(0, 0)),
    AddBias(in_features=64),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    AddBias(in_features=64),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3))
    # AddBias(in_features=3),
    # nn.Conv2d(3, 3, (1, 1)),

)


#
# decoder = nn.Sequential(
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.ConvTranspose2d(512, 256, (3, 3)),
#     nn.ReLU(),
#     nn.MaxUnpool2d((2, 2), (2, 2),(0, 0)),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.ConvTranspose2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.ConvTranspose2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.ConvTranspose2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.ConvTranspose2d(256, 128, (3, 3)),
#     nn.ReLU(),
#     nn.MaxUnpool2d((2, 2), (2, 2),(0, 0)),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.ConvTranspose2d(128, 128, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.ConvTranspose2d(128, 64, (3, 3)),
#     nn.ReLU(),
#     nn.MaxUnpool2d((2, 2), (2, 2),(0, 0)),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.ConvTranspose2d(64, 64, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.ConvTranspose2d(64, 3, (3, 3)),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.ConvTranspose2d(3, 3, (1, 1)),
#
# )



vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True, return_indices=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True, return_indices=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True,return_indices=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


decoder_mini_path = "models/vgg_deconv_mini.pth"


class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:8])  # input -> max_pool1
        self.enc_2 = nn.Sequential(*enc_layers[8:15])  # max_pool1 -> max_pool2
        self.enc_3 = nn.Sequential(*enc_layers[15:28])  # max_pool2 -> max_pool3
        self.enc_4 = nn.Sequential(*enc_layers[28:31])  # max_pool3 -> relu4_1

        self.dec_layers  = list(decoder.children())
        self.decoder = decoder

        self.decoder_mini = self.dec_layers[-7:]
        self.encoder_mini = nn.Sequential(*enc_layers[:6])
        self.mse_loss = nn.MSELoss()

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        switches = []
        for i in range(3):
            input, switch = getattr(self, 'enc_{:d}'.format(i + 1))(input)
            switches.append(switch)
        input = getattr(self, 'enc_{:d}'.format(4))(input)
        return input, switches

    def decode(self, input, switches):
        switch_index =2
        unpool_layers= [4,21,30]
        for layer_index,layer in enumerate(self.dec_layers):
            if layer_index in unpool_layers:
                input = layer(input, switches[switch_index])
                switch_index -=1
            else:
                input = layer(input)
        return input

    def encode_mask_layer(self, input, layer_index, mask):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
            if layer_index ==(i+1):
                input = input.data.masked_fill_(mask, 0.)
        return input


    def calc_content_loss(self, input, target):
        assert (input.data.size() == target.data.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.data.size() == target.data.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style):
        style_feats = self.encode_with_intermediate(style)
        t = adain(self.encode(content), style_feats[-1])

        g_t = self.decoder(Variable(t.data, requires_grad=True))
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s

    def encode_from_layer(self, input, layer_index):
        for i in range(layer_index, 4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def get_layer_encoding(self, input, layer_index):
        for i in range(layer_index):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def get_image_from_encoder_layer(self, input_variable, layer_index, amplitude=1.0):
        encoding = self.encode_from_layer(input_variable, layer_index).mul(amplitude)
        image = self.decoder(encoding)
        return image

    def get_channel_activation(self, input, mask, layer, amplitude):
        input = self.encode_mask_layer(input, layer, mask)
        #encoding = self.encode(input_variable)

        input = input.mul(amplitude)
        # encoding_numpy =encoding.cpu().data.numpy()[0]
        # masked_encoding = np.zeros_like(encoding_numpy, dtype=np.float32)
        # masked_encoding[channel,:,:] = encoding_numpy[channel,:,:]*amplitude
        #
        # masked_variable =  torch.from_numpy(masked_encoding)
        # masked_variable = masked_variable.cuda()
        # masked_variable = Variable(masked_variable, volatile=True).unsqueeze(0)

        input = self.decoder(input)

        max = input.max()
        min = input.min()
        input = input.sub(min).div(max-min+0.000001)
        return input

    def test_pass(self, input, amplitude=1.0):
        input, switches = self.encode(input)
        #input = input.mul(50.)

        input = self.decode(input, switches)
        return input

    def mini_encode(self, input):
        for layer in self.encoder_mini:
            input = layer(input)
        return input

    def mini_decode(self, input):
        for layer in self.decoder_mini:
            input = layer(input)
        return input

    def mini_pass(self, input, amplitude=1.0):
        input = self.mini_encode(input)
        # input = input.mul(50.)

        input = self.mini_decode(input)
        return input