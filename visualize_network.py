import sys
import numpy as np
from collections import OrderedDict
import torch
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QSlider,QVBoxLayout, QLabel, QDial, QWidget, QHBoxLayout, QComboBox, QGridLayout, QPushButton, QFileDialog
import net
import torch.nn as nn
from torch.autograd import Variable



decoder_path = "models/decoder.pth"
vgg_path = "models/vgg_normalised.pth"


def init_torch():
    torch.cuda.set_device(0)


class NeuralNetworkManager():
    def __init__(self, input_dim):
        self.input_dimension = input_dim
        decoder = net.decoder
        vgg = net.vgg

        self.generate_model_dict(vgg, self.input_dimension)

        vgg.load_state_dict(torch.load(vgg_path))
        vgg = nn.Sequential(*list(vgg.children())[:31])
        decoder.load_state_dict(torch.load(decoder_path))
        self.net = net.Net(vgg, decoder)
        self.net.cuda()


    def set_model_dict(self, model_dict):
        self.model_dict = model_dict

    def set_mask(self, mask_list):
        self.masks = []
        for layer_index, conv_layer in enumerate(self.conv_dict.values()):
            _, out_channels, _, _, resolution = conv_layer
            mask_array = np.zeros( (out_channels, resolution, resolution) ).astype(np.float32)
            select = np.where(mask_list[layer_index]==1.)
            mask_array[ select, :, :] = 1.
            mask_tensor = torch.from_numpy(mask_array).cuda()
            mask_variable = Variable(mask_tensor.unsqueeze(0), volatile=True)

            self.masks.append(mask_variable)

    def set_input(self, input):
        input = np.transpose(input, (2, 1, 0))
        input_tensor = torch.from_numpy(input)
        input_tensor = input_tensor.cuda()
        self.input_variable = Variable(input_tensor.unsqueeze(0), volatile=True)


    def encode_decode(self):
        image = self.net.decode(self.net.encode(self.input_variable))
        frame = image.cpu().data.numpy()[0]
        frame = np.transpose(frame, (1, 2, 0))
        return frame

    def encode_decode_masked(self):

        encoding = self.net.encode_with_masking(self.input_variable, self.masks)
        image = self.net.decode(encoding)
        frame = image.cpu().data.numpy()[0]
        frame = np.transpose(frame, (1, 2, 0))
        return frame

    def generate_model_dict(self, network, input_resolution):

        layer_dict = OrderedDict()
        conv_dict = OrderedDict()
        res = input_resolution
        for i, layer in enumerate(network.children()):
            layer_name = "{}-{}".format(i, layer.__repr__())
            if "Conv" in layer_name:
                layer_dict[layer_name] = [layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride]
                conv_dict[layer_name] = [layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride, res]
            elif "ReLU" in layer_name:
                layer_dict[layer_name] = []
            elif "MaxPool" in layer_name:
                layer_dict[layer_name] = [layer.kernel_size, layer.stride]
                res = int(res/2)
            else:
                layer_dict[layer_name] = []
        self.layer_dict = layer_dict
        self.conv_dict =conv_dict




class NeuronGroup(pg.GraphicsObject):
    def __init__(self, layer, x, y):

        pg.GraphicsObject.__init__(self)
        self.x = x
        self.y =y
        self.layer = layer
        self.selected = True
        self.generatePicture()
        self.neurons = []

    def set_neurons(self, neuron_list):
        self.neurons= neuron_list

    def click(self):
        self.selected = not self.selected
        for neuron in self.neurons:
            neuron.set_active(self.selected)

        self.generatePicture()
        self.update()

    def generatePicture(self):
        ## pre-computing a QPicture object allows paint() to run much more quickly,
        ## rather than re-drawing the shapes every time.
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setPen(pg.mkPen('w'))

        if self.selected:
            p.setBrush(pg.mkBrush('g'))
        else:
            p.setBrush(pg.mkBrush('r'))
        p.drawEllipse(self.x, self.y, 25, 25)
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        ## boundingRect _must_ indicate the entire area that will be drawn on
        ## or else we will get artifacts and possibly crashing.
        ## (in this case, QPicture does all the work of computing the bouning rect for us)

        return QtCore.QRectF(self.picture.boundingRect())

    def getLayer(self):
        return self.layer

class NeuronRect(pg.GraphicsObject):
    def __init__(self, layer, index, x, y):
        pg.GraphicsObject.__init__(self)
        self.x =x
        self.y =y
        self.layer = layer
        self.index=  index
        self.selected = True
        self.generatePicture()

    def generatePicture(self):
        ## pre-computing a QPicture object allows paint() to run much more quickly,
        ## rather than re-drawing the shapes every time.
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setPen(pg.mkPen('w'))

        if self.selected:
            p.setBrush(pg.mkBrush('g'))
        else:
            p.setBrush(pg.mkBrush('r'))
        p.drawRect(self.x,self.y,10,10)
        p.end()

    def set_active(self, active):
        self.selected = active
        self.generatePicture()
        self.update()

    def getLayerIndex(self):
        return self.layer, self.index

    def click(self):
        self.selected = not self.selected
        self.generatePicture()
        self.update()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        ## boundingRect _must_ indicate the entire area that will be drawn on
        ## or else we will get artifacts and possibly crashing.
        ## (in this case, QPicture does all the work of computing the bouning rect for us)

        return QtCore.QRectF(self.picture.boundingRect())


class MainWindow(QWidget):
    def __init__(self, input_dim):

        self.resolution =input_dim
        super(MainWindow, self).__init__()
        self.glw = pg.GraphicsLayoutWidget()

        self.noise_view = self.glw.addViewBox()
        self.noise_view.setAspectLocked()
        self.noise_img = pg.ImageItem()
        self.noise_view.addItem(self.noise_img)

        self.model_view = self.glw.addViewBox()
        self.model_view.setAspectLocked()
        self.output_view = self.glw.addViewBox()
        self.output_view.setAspectLocked()
        self.output_img = pg.ImageItem()
        self.output_view.addItem(self.output_img)


        self.glw.scene().sigMouseClicked.connect(self.onClick)
        #self.glw.sigSceneMouseMoved.connect(self.onClick)


        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.glw)

        self.nnm = NeuralNetworkManager(input_dim)

        self.set_model(self.nnm.conv_dict)

        self.create_noise()
        # self.graphicsLayout = pg.GraphicsLayoutWidget()
        # self.setLayout(self.graphicsLayout.ci)

    def set_model(self, modelDict):
        self.model = modelDict
        self.layers = []
        for _, out_channels, _, _, _ in self.model.values():
            self.layers.append(np.ones((out_channels)))

        self.nnm.set_mask(self.layers)


    def create_noise(self):
        input = np.random.rand( self.resolution, self.resolution,3).astype(np.float32)
        self.noise_img.setImage(input)
        self.nnm.set_input(input)

    def visualize_model(self):
        column_step = 0
        numLayers = len(self.model.items())
        self.rects = []

        for layerIndex, (key,layer ) in enumerate(self.model.items()):
            rects =[]
            print("Layer {}/{}".format(layerIndex,numLayers))
            if layerIndex >4 :continue
            in_channels, out_channels, kernel_size, stride, _ = layer
            neurons = []

            col_start = column_step
            for i in range(out_channels):
                height= i%32
                if (i%32) ==0:
                    column_step+=15

                neuron= NeuronRect(layerIndex, i, column_step, height *15)
                neurons.append(neuron)
                self.model_view.addItem(neuron)

            neuron_group = NeuronGroup(layerIndex,col_start + (column_step-col_start)/2 , 32*15+10)
            neuron_group.set_neurons(neurons)
            self.model_view.addItem(neuron_group)
            column_step +=20

    def onClick(self, event):
        items = self.glw.scene().items(event.scenePos())
        for x in items:
            if isinstance(x, NeuronRect):
                x.click()
                layer, index = x.getLayerIndex()
                self.layers[layer][index]= 1. if x.selected else 0.
                self.nnm.set_mask(self.layers)
                self.update()

            if isinstance(x, NeuronGroup):
                x.click()
                layer = x.getLayer()
                self.layers[layer] = np.ones_like(self.layers[layer]) if x.selected else np.zeros_like(self.layers[layer])

                self.nnm.set_mask(self.layers)
                self.update()


            if x is self.noise_view:
                self.create_noise()
                self.update()

    def update(self):
        img = self.nnm.encode_decode_masked()
        self.output_img.setImage(img)




if __name__ == "__main__":

    application = QtGui.QApplication([])


    mw = MainWindow(256)
    mw.visualize_model()
    mw.show()



    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()