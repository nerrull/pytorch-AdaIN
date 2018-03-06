import torch.nn as nn
from torch.autograd import Variable

from function import adaptive_instance_normalization as adain
from function import calc_mean_std
import numpy as np
import torch

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)


vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
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
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
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




class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())[:31]
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_layers = enc_layers
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()


    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def full_encode(self, input):
        for layer in self.enc_layers:
            input = layer(input)
        return input

    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def encode_mask_layer(self, input, layer_index, mask):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
            if layer_index ==(i+1):
                total = torch.sum(input.data,1)
                input.data = input.data.masked_fill_(mask, 0.)
        return input


    def encode_mask_layer_sum_activations(self, input, layer_index, neuron_index):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
            if layer_index ==(i+1):
                total = torch.sum(input.data,1)
                max = torch.max(total)
                neuron_activation =input.data[0, neuron_index, :, :]
                max_value = torch.max(neuron_activation)/2 +0.0001
                neuron_activation = neuron_activation*max/max_value
                input_shape = input.shape
                del max_value, max, total, input
                input = Variable(torch.zeros(input_shape)).cuda()
                input[0, neuron_index, :, :] = neuron_activation
                del neuron_activation
        return input

    # def encode_mask_layer_sum_activations(self, input, mask_layer_index, neuron_index):
    #
    #     for layer_index, layer in enumerate(self.enc_layers):
    #         input = getattr(self, 'enc_{:d}'.format(layer_index + 1))(input)
    #         if mask_layer_index ==(layer_index+1):
    #             total = torch.sum(input.data,1)
    #             #total =input.data[0, neuron_index, :, :]
    #             masked_values = Variable(torch.zeros(input.shape)).cuda()
    #             masked_values[0, neuron_index, :, :] = total
    #             input = masked_values
    #     return input


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

    def decode(self,input):
        return self.decoder(input)


    def encode_with_masking(self, input, masks):

        conv_index =0
        for layer in self.enc_layers:
            input = layer(input)
            if "Conv" in layer.__repr__():
                input = input.mul(masks[conv_index])
                conv_index+=1
        return input

    def get_channel_activation(self, input,  layer, amplitude, neuron_index):
        #input = self.encode_mask_layer(input, layer, mask)
        input = self.encode_mask_layer_sum_activations(input, layer, neuron_index)

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



    def decode_mask_layer_sum_activations(self, input, mask_layer_index, neuron_index):
        num_layers = len(list(self.decoder.children()))
        for layer_index, layer in enumerate(self.decoder.children()):
            if  (num_layers -layer_index) ==mask_layer_index:
                print( "Layer {} -{}".format(layer_index, layer.__repr__()))

                # if ("Conv" not in layer.__repr__()):
                #     print( "Layer {}, sepecified but it isn't a conv layer it is {}".format(layer_index, layer.__repr__()))
                #     raise  Exception
                total = torch.sum(input.data, 1)
                max = torch.max(total)
                neuron_activation = input.data[0, neuron_index, :, :]
                max_value = torch.max(neuron_activation) / 2 + 0.0001
                neuron_activation = neuron_activation * max / max_value
                input_shape = input.shape
                del max_value, max, total, input
                input = Variable(torch.zeros(input_shape)).cuda()
                input[0, neuron_index, :, :] = neuron_activation
            input = layer(input)
        return input


    def get_decoder_channel_activation(self, input, layer, amplitude, neuron_index):
        #input = self.encode_mask_layer(input, layer, mask)
        #input = self.encode(input)
        input = self.encode_mask_layer_sum_activations(input, layer, neuron_index)
        # encoding_numpy =encoding.cpu().data.numpy()[0]
        # masked_encoding = np.zeros_like(encoding_numpy, dtype=np.float32)
        # masked_encoding[channel,:,:] = encoding_numpy[channel,:,:]*amplitude
        #
        # masked_variable =  torch.from_numpy(masked_encoding)
        # masked_variable = masked_variable.cuda()
        # masked_variable = Variable(masked_variable, volatile=True).unsqueeze(0)

        input = self.decode_mask_layer_sum_activations(input, layer, neuron_index)

        max = input.max()
        min = input.min()
        input = input.sub(min).div(max-min+0.000001)
        return input