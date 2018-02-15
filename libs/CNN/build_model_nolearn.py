import os
import sys
import signal
import time
import shutil
import numpy as np
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, FeaturePoolLayer, BatchNormLayer, prelu
from lasagne.layers import Conv3DLayer, MaxPool3DLayer, Pool3DLayer, batch_norm
from lasagne import nonlinearities, objectives, updates
from lasagne.nonlinearities import softmax, rectify
from nolearn.lasagne import NeuralNet, BatchIterator, TrainSplit
from nolearn.lasagne.handlers import SaveWeights
from nolearn_utils.hooks import SaveTrainingHistory, PlotTrainingHistory, EarlyStopping
import warnings
warnings.simplefilter("ignore")

class Rotate_batch_Iterator(BatchIterator):
    """
    handle class for on-the-fly data augmentation on batches.
    Applying 90,180 and 270 degrees rotations and flipping
    """
    def transform(self, Xb, yb):
        Xb, yb = super(Rotate_batch_Iterator, self).transform(Xb, yb)

        # Flip a given percentage of the images at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        x_da = Xb[indices]

        # apply rotation to the input batch
        rotate_90 = x_da[:,:,:,::-1,:].transpose(0,1,2,4,3)
        rotate_180 = rotate_90[:,:,:,::-1,:].transpose(0,1,2,4,3)

        # apply flipped versions of rotated patches
        rotate_0_flipped = x_da[:,:,:,:,::-1]
        rotate_180_flipped = rotate_180[:,:,:,:,::-1]

        augmented_x = np.stack([rotate_180,
                                rotate_0_flipped,
                                rotate_180_flipped],
                                axis=1)

        # select random indices from computed transformations
        #r_indices = np.random.randint(0,7,size=augmented_x.shape[0])
        r_indices = np.random.randint(0,3,size=augmented_x.shape[0])

        Xb[indices] = np.stack([augmented_x[i,r_indices[i],:,:,:,:] for i in range(augmented_x.shape[0])])

        return Xb, yb


def cascade_model(options):
    """
    3D cascade model using Nolearn and Lasagne

    Inputs:
    - model_options:
    - weights_path: path to where weights should be saved

    Output:
    - nets = list of NeuralNets (CNN1, CNN2)
    """

    # model options
    num_channels = len(options['modalities'])
    train_split_perc = options['train_split']
    num_epochs = options['max_epochs']
    max_epochs_patience = options['patience']
    ps = options['patch_size'][0]

    # save model to disk to re-use it. Create an experiment folder
    # organize experiment
    if not os.path.exists(os.path.join(options['weight_paths'], options['experiment'])):
        os.mkdir(os.path.join(options['weight_paths'], options['experiment']))
    if not os.path.exists(os.path.join(options['weight_paths'], options['experiment'], 'nets')):
        os.mkdir(os.path.join(options['weight_paths'], options['experiment'], 'nets'))


    # --------------------------------------------------
    # first model
    # --------------------------------------------------
    layer1 = InputLayer(name='in', shape=(None, num_channels, ps, ps, ps))
    layer1 = prelu(batch_norm(Conv3DLayer(layer1, name='conv1_1', num_filters=32, filter_size=3, pad='same'), name = 'BN1'), name='p_relu1')
    layer1 = prelu(batch_norm(Conv3DLayer(layer1, name='conv1_2', num_filters=32, filter_size=3, pad='same'), name = 'BN2'), name='p_relu2')
    layer1 = Pool3DLayer(layer1,  name='avgpool_1', mode='max', pool_size=2, stride=2)
    layer1 = prelu(batch_norm(Conv3DLayer(layer1, name='conv2_1', num_filters=64, filter_size=3, pad='same'), name = 'BN3'), name = 'p_relu3')
    layer1 = prelu(batch_norm(Conv3DLayer(layer1, name='conv2_2', num_filters=64, filter_size=3, pad='same'), name = 'BN4'), name = 'p_relu4')
    layer1 = Pool3DLayer(layer1,  name='avgpoo2_1', mode='max', pool_size=2, stride=2)
    layer1 = DropoutLayer(layer1, name = 'l1drop', p=0.5)
    layer1 = prelu(DenseLayer(layer1, name='d_1', num_units = 256), name = 'p_relu_fn1')
    layer1 = DropoutLayer(layer1, name = 'l2drop', p=0.5)
    layer1 = prelu(DenseLayer(layer1, name='d_2', num_units = 128), name = 'p_relu_fn2')
    layer1 = DropoutLayer(layer1, name = 'l3drop', p=0.5)
    layer1 = prelu(DenseLayer(layer1, name='d_3', num_units = 64), name = 'p_relu_fn3')
    layer1 = DenseLayer(layer1, name = 'out', num_units = 2, nonlinearity=nonlinearities.softmax)

    # save weights
    net_model = 'model_1'
    net_weights = os.path.join(options['weight_paths'], options['experiment'], 'nets',  net_model + '.pkl' )
    net_history  = os.path.join(options['weight_paths'], options['experiment'], 'nets', net_model + '_history.pkl')

    net1 =  NeuralNet(
        layers= layer1,
        objective_loss_function=objectives.categorical_crossentropy,
        batch_iterator_train=Rotate_batch_Iterator(batch_size=128),
        update = updates.adadelta,
        on_epoch_finished=[
            SaveWeights(net_weights, only_best=True, pickle=False),
            SaveTrainingHistory(net_history),
            EarlyStopping(patience=max_epochs_patience)],
        verbose= options['net_verbose'],
        max_epochs= num_epochs,
        train_split=TrainSplit(eval_size= train_split_perc),
    )

    # --------------------------------------------------
    # second model
    # --------------------------------------------------

    layer2 = InputLayer(name='in', shape=(None, num_channels, ps, ps, ps))
    layer2 = prelu(batch_norm(Conv3DLayer(layer2, name='conv1_1', num_filters=32, filter_size=3, pad='same'), name = 'BN1'), name='p_relu1')
    layer2 = prelu(batch_norm(Conv3DLayer(layer2, name='conv1_2', num_filters=32, filter_size=3, pad='same'), name = 'BN2'), name='p_relu2')
    layer2 = Pool3DLayer(layer2,  name='avgpool_1', mode='max', pool_size=2, stride=2)
    layer2 = prelu(batch_norm(Conv3DLayer(layer2, name='conv2_1', num_filters=64, filter_size=3, pad='same'), name = 'BN3'), name = 'p_relu3')
    layer2 = prelu(batch_norm(Conv3DLayer(layer2, name='conv2_2', num_filters=64, filter_size=3, pad='same'), name = 'BN4'), name = 'p_relu4')
    layer2 = Pool3DLayer(layer2,  name='avgpoo2_1', mode='max', pool_size=2, stride=2)
    layer2 = DropoutLayer(layer2, name = 'l1drop', p=0.5)
    layer2 = prelu(DenseLayer(layer2, name='d_1', num_units = 256), name = 'p_relu_fn1')
    layer2 = DropoutLayer(layer2, name = 'l2drop', p=0.5)
    layer2 = prelu(DenseLayer(layer2, name='d_2', num_units = 128), name = 'p_relu_fn2')
    layer2 = DropoutLayer(layer2, name = 'l3drop', p=0.5)
    layer2 = prelu(DenseLayer(layer2, name='d_3', num_units = 64), name = 'p_relu_fn3')
    layer2 = DenseLayer(layer2, name = 'out', num_units = 2, nonlinearity=nonlinearities.softmax)

    # save weights
    net_model = 'model_2'
    net_weights2 = os.path.join(options['weight_paths'], options['experiment'], 'nets',  net_model + '.pkl' )
    net_history2  = os.path.join(options['weight_paths'], options['experiment'], 'nets', net_model + '_history.pkl')

    net2 =  NeuralNet(
        layers= layer2,
        objective_loss_function=objectives.categorical_crossentropy,
        batch_iterator_train=Rotate_batch_Iterator(batch_size=128),
        update = updates.adadelta,
        on_epoch_finished=[
            SaveWeights(net_weights2, only_best=True, pickle=False),
            SaveTrainingHistory(net_history2),
            EarlyStopping(patience=max_epochs_patience)],
        verbose= options['net_verbose'],
        max_epochs= num_epochs,
        train_split=TrainSplit(eval_size= train_split_perc),
    )

    if options['full_train'] is False:

        if options['pretrained_model'] == 'baseline':
            # load default weights
            print "> CNN: Loading pretrained weights from the default configuration!"
            current_folder = os.path.dirname(os.path.abspath(__file__))
            net1_w_def = os.path.join(current_folder, 'defaults', 'baseline', 'nets', 'model_1.pkl')
            net2_w_def = os.path.join(current_folder, 'defaults', 'baseline', 'nets', 'model_2.pkl')
        else:
            # load default weights
            print "> CNN: Loading pretrained weights from the", options['pretrained_model'], "configuration"
            pretrained_model = os.path.join(options['weight_paths'], options['pretrained_model'],'nets')
            model = os.path.join(options['weight_paths'], options['experiment'])
            net1_w_def = os.path.join(model, 'nets', 'model_1.pkl')
            net2_w_def = os.path.join(model, 'nets', 'model_2.pkl')

            if not os.path.exists(model):
                shutil.copy(pretrained_model, model)
            else:
                shutil.copyfile(os.path.join(pretrained_model, 'model_1.pkl'), net1_w_def)
                shutil.copyfile(os.path.join(pretrained_model, 'model_2.pkl'), net2_w_def)

        try:
            net1.verbose = 0
            net1.load_params_from(net1_w_def)
            net2.verbose = 0
            net2.load_params_from(net2_w_def)
        except:
            print "> ERROR: The model", options['experiment'], 'selected does not contain a valid network model'
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)


    if options['load_weights'] is True:
        print "> CNN: loading weights from", options['experiment'], 'configuration'
        net1.verbose = 0
        net2.verbose = 0
        net1.load_params_from(net_weights)
        net2.load_params_from(net_weights2)

    net1.verbose = options['net_verbose']
    net2.verbose = options['net_verbose']

    return [net1, net2]


def define_training_layers(net, num_layers = None, number_of_samples = None):
    """
    Define the number of layers to train and freeze the rest

    inputs:
    - model: Neural network object net1 or net2
    - number of layers to retrain
    - nunber of training samples

    outputs
    - updated model
    """

    # use the nunber of samples to choose the number of layers to retrain
    if number_of_samples is not None:
        if number_of_samples < 10000:
            num_layers = 1
        elif number_of_samples < 100000:
            num_layers = 2
        else:
            num_layers = 3

    # check num_layers is set. set default value instead
    if (num_layers == 'None') and (number_of_samples is None):
        num_layers = 1


    print "> CNN: freezing the first", 11 - num_layers, 'layers'
    # freeze parameters
    net.initialize()
    net.layers_['conv1_1'].params[net.layers_['conv1_1'].W].remove("trainable")
    net.layers_['conv1_2'].params[net.layers_['conv1_2'].W].remove("trainable")
    net.layers_['conv2_1'].params[net.layers_['conv2_1'].W].remove("trainable")
    net.layers_['conv2_2'].params[net.layers_['conv2_2'].W].remove("trainable")
    net.layers_['p_relu1'].params[net.layers_['p_relu1'].alpha].remove("trainable")
    net.layers_['p_relu2'].params[net.layers_['p_relu2'].alpha].remove("trainable")
    net.layers_['p_relu3'].params[net.layers_['p_relu3'].alpha].remove("trainable")
    net.layers_['p_relu4'].params[net.layers_['p_relu4'].alpha].remove("trainable")

    if num_layers == 1:
        net.layers_['d_1'].params[net.layers_['d_1'].W].remove("trainable")
        net.layers_['p_relu_fn1'].params[net.layers_['p_relu_fn1'].alpha].remove("trainable")
        net.layers_['d_2'].params[net.layers_['d_2'].W].remove("trainable")
        net.layers_['p_relu_fn2'].params[net.layers_['p_relu_fn2'].alpha].remove("trainable")
    if num_layers == 2:
        net.layers_['d_1'].params[net.layers_['d_1'].W].remove("trainable")
        net.layers_['p_relu_fn1'].params[net.layers_['p_relu_fn1'].alpha].remove("trainable")
    if num_layers == 3:
        pass

    #if num_layers > 0:
    #     net.layers_['d_1'].params[net.layers_['d_1'].W].remove("trainable")
    #     net.layers_['p_relu_fn1'].params[net.layers_['p_relu_fn1'].alpha].remove("trainable")

    return net
