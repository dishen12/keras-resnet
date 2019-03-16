from __future__ import division

import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Dropout,
    Reshape,
    GlobalAveragePooling2D
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.layers.core import Layer
try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations

import tensorflow as tf
slim = tf.contrib.slim

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)

def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f

def conv_layer(**conv_params):
    """
    build a conv layer with batchnormalization and relu activatation 
    """
    filters = conv_params["filters"]
    kernel_size = conv_params.setdefault("kernel_size",3)
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    def f(input):
        activation = _bn_relu(input)
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)
        return _bn_relu(conv)
    return f

def build_vgg(cfg=[128, 128, 'M', 2*128, 2*128, 'M', 4*128, 4*128, 'M', (8*128, 0), 'M']):
    """
    build a vgg7 network
    """
    def f(input):
        layer_output = [] #the ith output is stored in ith of layer_output
        layer_output.append(input)
        if(len(cfg)==0): return input
        for i,v in enumerate(cfg):
            if(v == "M"):
                layer_output.append(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer_output[i]))
            elif(v == "M3"):
                layer_output.append(MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(layer_output[i]))
            elif(isinstance(v,tuple) and v[1]==1):
                padding = "same"
                filters = v[0]
                layer_output.append(conv_layer(filters=filters,kernel_size=1,padding=padding)(layer_output[i]))
            else:
                padding = "valid" if(isinstance(v,tuple) and v[1]==0) else "same"
                filters = v[0] if isinstance(v,tuple) else v 
                layer_output.append(conv_layer(filters=filters,padding=padding)(layer_output[i]))
            #print("shape is !!!!!!!!!!!!!!",layer_output[i].shape,"i is ",i)
        return layer_output[-1]
    
    return f

def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters, init_strides=(1, 1), is_first_block_of_first_layer=False)(input)
        return input

    return f

def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f

def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f

def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    #print("input is ",input,"shape of input is :",input.shape)
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    #print("next",input_shape,residual_shape)
    if(input_shape[ROW_AXIS]>residual_shape[ROW_AXIS] and input_shape[COL_AXIS]>residual_shape[COL_AXIS]):
        stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
        stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
        #print(stride_width,stride_width)
        equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

        shortcut = input
        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                              kernel_size=(1, 1),
                              strides=(stride_width, stride_height),
                              padding="valid",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(0.0001))(input)
    else:
        stride_width = int(round(residual_shape[ROW_AXIS] / input_shape[ROW_AXIS]))
        stride_height = int(round(residual_shape[COL_AXIS] / input_shape[COL_AXIS]))
        #print(stride_width,stride_width)
        equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

        shortcut = input
        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                              kernel_size=(1, 1),
                              strides=(stride_width, stride_height),
                              padding="valid",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])

def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3

def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier

class AddWeight(Layer):
    def __init__(self,weight_size,name,**kwargs):
        self.weight_size = 3
        self.name = name
        super(AddWeight, self).__init__(**kwargs)
    def build(self,input_shape):
        self.w1 = K.variable(initializations.RandomUniform(minval=-0.05, maxval=0.05, seed=None)((1,)), name='{}_w1'.format(self.name))
        self.w2 = K.variable(initializations.RandomUniform(minval=-0.05, maxval=0.05, seed=None)((1,)), name='{}_w2'.format(self.name))
        self.w3 = K.variable(initializations.RandomUniform(minval=-0.05, maxval=0.05, seed=None)((1,)), name='{}_w3'.format(self.name))
        self.trainable_weights = {self.w1,self.w2,self.w3}
        
    def call(self,x,mask=None):
        out = self.w1*x[0]+self.w2*x[1]+self.w3*x[2]
        #print("out shape:",out.shape)
        return out
    
    def compute_output_shape(self,input_shape):
        return (None,1,1,2048)
        
def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f

        
class vggBuilder(object):
    @staticmethod
    def vgg7(input_shape,num_outputs):
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])
        input = Input(input_shape)
        vgg7_out = build_vgg()(input)
        flatten = Flatten()(vgg7_out)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(flatten)
        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def vggA(input_shape,num_outputs):
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")
        print("vggA is there!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])
        input = Input(input_shape)
        
        vgg11_out = build_vgg(cfg=[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'])(input)
        #flatten = Flatten()(vgg7_out)
        pool2 = GlobalAveragePooling2D()(vgg11_out)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(pool2)
        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def vggB(input_shape,num_outputs):
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])
        input = Input(input_shape)
        
        vgg11_out = build_vgg(cfg=[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'])(input)
        #flatten = Flatten()(vgg7_out)
        pool2 = GlobalAveragePooling2D()(vgg11_out)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(pool2)
        model = Model(inputs=input, outputs=dense)
        return model
    
    @staticmethod
    def vggC(input_shape,num_outputs):
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])
        input = Input(input_shape)
        
        vgg11_out = build_vgg(cfg=[64,64,'M',128,128,'M',256,256,(256,1),'M',512,512,(512,1),'M',512,512,(512,1),'M'])(input)
        #flatten = Flatten()(vgg7_out)
        pool2 = GlobalAveragePooling2D()(vgg11_out)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(pool2)
        model = Model(inputs=input, outputs=dense)
        return model
    
    @staticmethod
    def vggD(input_shape,num_outputs):
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])
        input = Input(input_shape)
        
        vgg11_out = build_vgg(cfg=[64,64,'M',128,128,'M',256,256,(256,1),'M',512,512,512,'M',512,512,512,'M'])(input)
        #flatten = Flatten()(vgg7_out)
        pool2 = GlobalAveragePooling2D()(vgg11_out)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(pool2)
        model = Model(inputs=input, outputs=dense)
        return model
    
    @staticmethod
    def build_cross1(input_shape, num_outputs, block_fn, repetitions,only_last=False,cfg=[128, 128, 'M', 2*128, 2*128, 'M', 4*128, 4*128]):
        """Builds a custom ResNet like architecture.

        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        #K.set_image_dim_ordering('tf')
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        """
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)
        """
        vgg_out = build_vgg(cfg=cfg)(input)
        init = vgg_out
        
        #first branch
        block = init
        filters = 64
        branch_list_1 = []
        for i, r in enumerate(repetitions):
            if(r==0): continue
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2
            branch_list_1.append(block)
        block_branch_1 = block
        
        #second branch
        block = init
        filters = 64
        branch_list_2 = []
        for i, r in enumerate(repetitions):
            if(r==0): continue
            if(i==0):
                block = _shortcut(branch_list_1[-1], block)
                #block = block
            else:
                block = _shortcut(branch_list_1[i-1], block)
            """
            elif(i==1):
                block = _shortcut(branch_list_1[i-1], block)
            elif(i==2):
                block = _shortcut(branch_list_1[i-1], block)
            elif(i==3):
                block = _shortcut(branch_list_1[i-1], block)
            """
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2
            branch_list_2.append(block)
        block_branch_2 = block
        
        #third branch
        block = init
        filters = 64
        branch_list_3 = []
        for i, r in enumerate(repetitions):
            if(r==0): continue
            if(i==0):
                block = _shortcut(branch_list_2[-1], block)
                #block =block
            else:
                block = _shortcut(branch_list_2[i-1], block)
            """
            elif(i==1):
                block = _shortcut(branch_list_2[i-1], block)
            elif(i==2):
                block = _shortcut(branch_list_2[i-1], block)
            elif(i==3):
                block = _shortcut(branch_list_2[i-1], block)
            """
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2
            branch_list_3.append(block)
        block_branch_3 = block    
        #print("init block size",block_branch_3)
        #merge
        if(only_last):
            block = block_branch_3 
        else:
            block = AddWeight(weight_size=3,name="add_weight")([block_branch_1,block_branch_2,block_branch_3])
        # Last activation
        block = _bn_relu(block)

        # Classifier block
        #print("block is ",block)
        #block_shape = K.int_shape(block)
        block_shape = block.shape
        #print("block_shape",block_shape)
        #pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),strides=(1, 1))(block)
        #print(ROW_AXIS,COL_AXIS,block_shape[1])
        #pool2 = AveragePooling2D(pool_size=(8,8),strides=(1 , 1))(block)
        pool2 = GlobalAveragePooling2D()(block)
        #print("pool2",pool2)
        #flatten1 = Flatten()(pool2)
        #flatten1 = Reshape((131072,))(pool2)
        #print('flatten1',flatten1.shape)
#         fc1 = Dense(4096, activation='relu')(flatten1)
#         drop = Dropout(0.5)(fc1)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                       activation="softmax")(pool2)
        #print(pool2,flatten1,dense)
        model = Model(inputs=input, outputs=dense)
        #print(model.summary())
        return model

    @staticmethod    
    def build_three_branch_18(input_shape, num_outputs):
        return vggBuilder.build_cross1(input_shape, num_outputs, bottleneck, [2,2,2,2])
    @staticmethod                                  
    def build_three_branch_32(input_shape, num_outputs):
        return vggBuilder.build_cross1(input_shape, num_outputs, bottleneck, [3,4,6,3])
    @staticmethod
    def build_three_branch_50(input_shape, num_outputs):
        return vggBuilder.build_cross1(input_shape, num_outputs, bottleneck, [3,4,6,3])
    @staticmethod
    def build_three_branch_101(input_shape, num_outputs):
        return vggBuilder.build_cross1(input_shape, num_outputs, bottleneck, [3,4,23,3])
    @staticmethod
    def build_three_branch_152(input_shape, num_outputs):
        return vggBuilder.build_cross1(input_shape, num_outputs, bottleneck, [3,8,36,3])
    @staticmethod    
    def build_three_branch_input(input_shape, num_outputs,input_layer=[2,2,2,2]):
        #print(input_layer)
        return vggBuilder.build_cross1(input_shape, num_outputs, bottleneck, input_layer)
    @staticmethod
    def build_three_branch_18_only_last(input_shape, num_outputs,only_last,cfg=[128, 128, 'M', 2*128, 2*128, 'M']):
        return vggBuilder.build_cross1(input_shape, num_outputs, bottleneck, [1,1,1,1])
    @staticmethod
    def build_three_branch_18_only_last_model1(input_shape, num_outputs,only_last,cfg=[128, 128, 'M']):
        return vggBuilder.build_cross1(input_shape, num_outputs, bottleneck, [1,1,1,1])
    @staticmethod
    def build_three_branch_18_only_last_model2(input_shape, num_outputs,only_last,cfg=[]):
        return vggBuilder.build_cross1(input_shape, num_outputs, bottleneck, [1,1,1,1])
    @staticmethod
    def build_three_branch_2222_only_last(input_shape, num_outputs,only_last):
        return vggBuilder.build_cross1(input_shape, num_outputs, bottleneck, [2,2,2,2])
    @staticmethod
    def buildVgg7(input_shape, num_outputs):
        return vggBuilder.vgg7(input_shape,num_outputs)