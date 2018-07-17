from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers.advanced_activations import PReLU as prelu
from keras.layers.normalization import BatchNormalization as BN
from keras import backend as K
from keras.models import Model
K.set_image_dim_ordering('th')

def get_network(options):
    """
    CNN model for MS lesion segmentation. Based on the model proposed on:

    Valverde, S. et al (2017). Improving automated multiple sclerosis lesion
    segmentation with a cascaded 3D convolutional neural network approach.
    NeuroImage, 155, 159-168. https://doi.org/10.1016/j.neuroimage.2017.04.034

    However, two additional fully-connected layers are added to increase
    the effective transfer learning
    """

    # model options
    channels = len(options['modalities'])

    net_input = Input(name='in1', shape=(channels,) + options['patch_size'])
    layer = Conv3D(filters=32, kernel_size=(3, 3, 3),
                   name='conv1_1',
                   activation=None,
                   padding="same")(net_input)
    layer = BN(name='bn_1_1', axis=1)(layer)
    layer = prelu(name='prelu_conv1_1')(layer)
    layer = Conv3D(filters=32,
                   kernel_size=(3, 3, 3),
                   name='conv1_2',
                   activation=None,
                   padding="same")(layer)
    layer = BN(name='bn_1_2', axis=1)(layer)
    layer = prelu(name='prelu_conv1_2')(layer)
    layer = MaxPooling3D(pool_size=(2, 2, 2),
                         strides=(2, 2, 2))(layer)
    layer = Conv3D(filters=64,
                   kernel_size=(3, 3, 3),
                   name='conv2_1',
                   activation=None,
                   padding="same")(layer)
    layer = BN(name='bn_2_1', axis=1)(layer)
    layer = prelu(name='prelu_conv2_1')(layer)
    layer = Conv3D(filters=64,
                   kernel_size=(3, 3, 3),
                   name='conv2_2',
                   activation=None,
                   padding="same")(layer)
    layer = BN(name='bn_2_2', axis=1)(layer)
    layer = prelu(name='prelu_conv2_2')(layer)
    layer = MaxPooling3D(pool_size=(2, 2, 2),
                         strides=(2, 2, 2))(layer)
    layer = Flatten()(layer)
    layer = Dropout(name='dr_d1', rate=0.5)(layer)
    layer = Dense(units=256,  activation=None, name='d1')(layer)
    layer = prelu(name='prelu_d1')(layer)
    layer = Dropout(name='dr_d2', rate=0.5)(layer)
    layer = Dense(units=128,  activation=None, name='d2')(layer)
    layer = prelu(name='prelu_d2')(layer)
    layer = Dropout(name='dr_d3', rate=0.5)(layer)
    layer = Dense(units=64,  activation=None, name='d3')(layer)
    layer = prelu(name='prelu_d3')(layer)
    net_output = Dense(units=2, name='out', activation='softmax')(layer)

    model = Model(inputs=[net_input], outputs=net_output)

    return model
