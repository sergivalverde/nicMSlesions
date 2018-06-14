import os
import signal
import time
import shutil
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from nets import get_network

# force data format to "channels first"
keras.backend.set_image_data_format('channels_first')


def transform(Xb, yb):
    """
    handle class for on-the-fly data augmentation on batches.
    Applying 90,180 and 270 degrees rotations and flipping
    """
    # Flip a given percentage of the images at random:
    bs = Xb.shape[0]
    indices = np.random.choice(bs, bs / 2, replace=False)
    x_da = Xb[indices]

    # apply rotation to the input batch
    rotate_90 = x_da[:, :, :, ::-1, :].transpose(0, 1, 2, 4, 3)
    rotate_180 = rotate_90[:, :, :, :: -1, :].transpose(0, 1, 2, 4, 3)

    # apply flipped versions of rotated patches
    rotate_0_flipped = x_da[:, :, :, :, ::-1]
    rotate_180_flipped = rotate_180[:, :, :, :, ::-1]

    augmented_x = np.stack([rotate_180,
                            rotate_0_flipped,
                            rotate_180_flipped],
                            axis=1)

    # select random indices from computed transformations
    r_indices = np.random.randint(0, 3, size=augmented_x.shape[0])

    Xb[indices] = np.stack([augmented_x[i,
                                        r_indices[i], :, :, :, :]
                            for i in range(augmented_x.shape[0])])

    return Xb, yb


def da_generator(x_train, y_train, batch_size=256):
    """
    Keras generator used for training with data augmentation. This generator
    calls the data augmentation function yielding training samples
    """
    num_samples = x_train.shape[0]
    while True:
        for b in range(0, num_samples, batch_size):
            x_ = x_train[b:b+batch_size]
            y_ = y_train[b:b+batch_size]
            x_, y_ = transform(x_, y_)
            yield x_, y_


def cascade_model(options):
    """
    3D cascade model using Nolearn and Lasagne

    Inputs:
    - model_options:
    - weights_path: path to where weights should be saved

    Output:
    - nets = list of NeuralNets (CNN1, CNN2)
    """

    # save model to disk to re-use it. Create an experiment folder
    # organize experiment
    if not os.path.exists(os.path.join(options['weight_paths'],
                                       options['experiment'])):
        os.mkdir(os.path.join(options['weight_paths'],
                              options['experiment']))
    if not os.path.exists(os.path.join(options['weight_paths'],
                                       options['experiment'], 'nets')):
        os.mkdir(os.path.join(options['weight_paths'],
                              options['experiment'], 'nets'))
    if options['debug']:
        if not os.path.exists(os.path.join(options['weight_paths'],
                                           options['experiment'],
                                           '.train')):
            os.mkdir(os.path.join(options['weight_paths'],
                                  options['experiment'],
                                  '.train'))

    # --------------------------------------------------
    # model 1
    # --------------------------------------------------

    model = get_network(options)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    if options['debug']:
        model.summary()

    # save weights
    net_model = 'model_1'
    net_weights_1 = os.path.join(options['weight_paths'],
                                 options['experiment'],
                                 'nets', net_model + '.hdf5')

    net1 = {}
    net1['net'] = model
    net1['weights'] = net_weights_1
    net1['history'] = None

    # --------------------------------------------------
    # model 2
    # --------------------------------------------------

    model2 = get_network(options)
    model2.compile(loss='categorical_crossentropy',
                   optimizer='adadelta',
                   metrics=['accuracy'])
    if options['debug']:
        model2.summary()

    # save weights
    net_model = 'model_2'
    net_weights_2 = os.path.join(options['weight_paths'],
                                 options['experiment'],
                                 'nets', net_model + '.hdf5')

    net2 = {}
    net2['net'] = model2
    net2['weights'] = net_weights_2
    net2['history'] = None

    # load predefined weights if transfer learning is selected

    if options['full_train'] is False:

        # load default weights
        print "> CNN: Loading pretrained weights from the", \
        options['pretrained_model'], "configuration"
        pretrained_model = os.path.join(options['weight_paths'], \
                                        options['pretrained_model'],'nets')
        model = os.path.join(options['weight_paths'],
                             options['experiment'])
        net1_w_def = os.path.join(model, 'nets', 'model_1.hdf5')
        net2_w_def = os.path.join(model, 'nets', 'model_2.hdf5')

        if not os.path.exists(model):
            shutil.copy(pretrained_model, model)
        else:
            shutil.copyfile(os.path.join(pretrained_model,
                                         'model_1.hdf5'),
                            net1_w_def)
            shutil.copyfile(os.path.join(pretrained_model,
                                         'model_2.hdf5'),
                            net2_w_def)
        try:
            net1['net'].load_weights(net1_w_def, by_name=True)
            net2['net'].load_weights(net2_w_def, by_name=True)
        except:
            print "> ERROR: The model", \
                options['experiment'],  \
                'selected does not contain a valid network model'
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)

    if options['load_weights'] is True:
        print "> CNN: loading weights from", \
            options['experiment'], 'configuration'
        print net_weights_1
        print net_weights_2

        net1['net'].load_weights(net_weights_1, by_name=True)
        net2['net'].load_weights(net_weights_2, by_name=True)

    return [net1, net2]


def define_training_layers(model, num_layers=1, number_of_samples=None):
    """
    Define the number of layers to train and freeze the rest

    inputs: - model: Neural network object net1 or net2 - number of
    layers to retrain - nunber of training samples

    outputs - updated model """

    # all layers are first set to non trainable
    net = model['net']
    for l in net.layers:
        l.trainable = False

    # use the nunber of samples to choose the number of layers to retrain
    if number_of_samples is not None:
        if number_of_samples < 10000:
            num_layers = 1
        elif number_of_samples < 100000:
            num_layers = 2
        else:
            num_layers = 3

    print "> CNN: re-training the last", num_layers, "layers"

    # set FC layers
    if num_layers == 1:
        net.get_layer('dr_d3').trainable = True
        net.get_layer('d3').trainable = True
        net.get_layer('prelu_d3').trainable = True
    if num_layers == 2:
        net.get_layer('dr_d3').trainable = True
        net.get_layer('d3').trainable = True
        net.get_layer('prelu_d3').trainable = True
        net.get_layer('dr_d2').trainable = True
        net.get_layer('d2').trainable = True
        net.get_layer('prelu_d2').trainable = True
    if num_layers == 3:
        net.get_layer('dr_d3').trainable = True
        net.get_layer('d3').trainable = True
        net.get_layer('prelu_d3').trainable = True
        net.get_layer('dr_d2').trainable = True
        net.get_layer('d2').trainable = True
        net.get_layer('prelu_d2').trainable = True
        net.get_layer('dr_d3').trainable = True
        net.get_layer('d3').trainable = True
        net.get_layer('prelu_d3').trainable = True

    net.compile(loss='categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])

    model['net'] = net
    return model


def fit_model(model, x_train, y_train, options, initial_epoch=0):
    """
    fit the cascaded model.

    """
    num_epochs = options['max_epochs']
    train_split_perc = options['train_split']
    batch_size = options['batch_size']

    # convert labels to categorical
    # y_train = keras.utils.to_categorical(y_train, len(np.unique(y_train)))
    y_train = keras.utils.to_categorical(y_train == 1,
                                         len(np.unique(y_train == 1)))

    # split training and validation
    perm_indices = np.random.permutation(x_train.shape[0])
    train_val = int(len(perm_indices)*train_split_perc)

    x_train_ = x_train[:train_val]
    y_train_ = y_train[:train_val]
    x_val_ = x_train[train_val:]
    y_val_ = y_train[train_val:]

    h = model['net'].fit_generator(da_generator(
        x_train_, y_train_,
        batch_size=batch_size),
        validation_data=(x_val_, y_val_),
        epochs=num_epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=x_train_.shape[0]/batch_size,
        verbose=options['net_verbose'],
        callbacks=[ModelCheckpoint(model['weights'],
                                   save_best_only=True,
                                   save_weights_only=True),
                   EarlyStopping(monitor='val_loss',
                                 min_delta=0,
                                 patience=options['patience'],
                                 verbose=0,
                                 mode='auto')])


    model['history'] = h

    if options['debug']:
        print "> DEBUG: loading best weights after training"

    model['net'].load_weights(model['weights'])

    return model
