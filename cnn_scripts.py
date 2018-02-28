
# ------------------------------------------------------------------------------------------------------------
#   MS lesion segmentation pipeline
# ---------------------------------
#   - incorporates:
#         - MRI identification
#         - registration
#         - skull stripping
#         - MS lesion segmentation training and testing using the CNN aproach
#           of Valverde et al (NI2017)
#
#  Sergi Valverde 2017
#  svalverde@eia.udg.edu
# ------------------------------------------------------------------------------------------------------------

import os
import sys
import platform
import time
import ConfigParser
from utils.preprocess import preprocess_scan
from utils.load_options import load_options, print_options
CURRENT_PATH = os.getcwd()
sys.path.append(os.path.join(CURRENT_PATH, 'libs'))


def get_config():
    """
    Get the CNN configuration from file
    """
    default_config = ConfigParser.SafeConfigParser()
    default_config.read(os.path.join(CURRENT_PATH, 'config', 'default.cfg'))
    user_config = ConfigParser.RawConfigParser()
    user_config.read(os.path.join(CURRENT_PATH, 'config', 'configuration.cfg'))

    # read user's configuration file
    options = load_options(default_config, user_config)
    options['tmp_folder'] = CURRENT_PATH + '/tmp'

    # set paths taking into account the host OS
    host_os = platform.system()
    if host_os == 'Linux':
        options['niftyreg_path'] = CURRENT_PATH + '/libs/linux/niftyreg'
        options['robex_path'] = CURRENT_PATH + '/libs/linux/ROBEX/runROBEX.sh'
        options['test_slices'] = 256
    elif host_os == 'Windows':
        options['niftyreg_path'] = os.path.normpath(
            os.path.join(CURRENT_PATH,
                         'libs',
                         'win',
                         'niftyreg'))

        options['robex_path'] = os.path.normpath(
            os.path.join(CURRENT_PATH,
                         'libs',
                         'win',
                         'ROBEX',
                         'runROBEX.bat'))
        options['test_slices'] = 256
    else:
        print "The OS system", host_os, "is not currently supported."
        exit()

    # print options when debugging
    if options['debug']:
        print_options(options)

    return options


def train_network(options):
    """
    Train the CNN network given the options passed as parameter
    """
    # set GPU mode from the configuration file.
    # So, this has to be updated before calling
    # the CNN libraries if the default config "~/.theanorc" has to be replaced.
    if options['mode'].find('cuda') == -1 and options['mode'].find('gpu') == -1:
        os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cpu,floatX=float32,optimizer=fast_compile'
    else:
        os.environ['THEANO_FLAGS']='mode=FAST_RUN,device='+options['mode'] +',floatX=float32,optimizer=fast_compile'

    from CNN.base import train_cascaded_model
    from CNN.build_model_nolearn import cascade_model

    scan_list = os.listdir(options['train_folder'])
    scan_list.sort()

    options['train_folder'] = os.path.normpath(options['train_folder'])
    for scan in scan_list:

        total_time = time.time()

        # --------------------------------------------------
        # move things to a tmp folder before starting
        # --------------------------------------------------

        options['tmp_scan'] = scan
        current_folder = os.path.join(options['train_folder'], scan)
        options['tmp_folder'] = os.path.normpath(os.path.join(current_folder,
                                                              'tmp'))
        # preprocess scan
        preprocess_scan(current_folder, options)

    # --------------------------------------------------
    # WM MS lesion training
    # - configure net and train
    # --------------------------------------------------

    seg_time = time.time()
    print "> CNN: Starting training session"
    # select training scans
    train_x_data = {f: {m: os.path.join(options['train_folder'], f, 'tmp', n)
                        for m, n in zip(options['modalities'],
                                        options['x_names'])}
                    for f in scan_list}
    train_y_data = {f: os.path.join(options['train_folder'],
                                    f,
                                    'tmp',
                                    options['ROI_name']) for f in scan_list}

    options['weight_paths'] = os.path.join(CURRENT_PATH, 'nets')
    options['load_weights'] = False

    # train the model for the current scan

    print "> CNN: training net with %d subjects" % (len(train_x_data.keys()))

    # --------------------------------------------------
    # initialize the CNN and train the classifier
    # --------------------------------------------------
    model = cascade_model(options)
    model = train_cascaded_model(model, train_x_data, train_y_data,  options)

    print "> INFO: training time:", round(time.time() - seg_time), "sec"
    print "> INFO: total pipeline time: ", round(time.time() - total_time), "sec"
    print "> INFO: All processes have been finished. Have a good day!"


def infer_segmentation(options):
    """
    Infer segmentation given the input options passed as parameters
    """

    # set GPU mode from the configuration file. So, this has to be updated before calling
    # the CNN libraries if the default config "~/.theanorc" has to be replaced.
    if options['mode'].find('cuda') == -1 and options['mode'].find('gpu') == -1:
        os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cpu,floatX=float32,optimizer=fast_compile'
    else:
        os.environ['THEANO_FLAGS']='mode=FAST_RUN,device='+options['mode'] +',floatX=float32,optimizer=fast_compile'

    from CNN.base import test_cascaded_model
    from CNN.build_model_nolearn import cascade_model

    # --------------------------------------------------
    # net configuration
    # take into account if the pretrained models have to be used
    # all images share the same network model
    # --------------------------------------------------
    options['full_train'] = True
    options['load_weights'] = True
    options['weight_paths'] = os.path.join(CURRENT_PATH, 'nets')
    options['net_verbose'] = 0
    model = cascade_model(options)

    # --------------------------------------------------
    # process each of the scans
    # - image identification
    # - image registration
    # - skull-stripping
    # - WM segmentation
    # --------------------------------------------------
    scan_list = os.listdir(options['test_folder'])
    scan_list.sort()

    for scan in scan_list:

        total_time = time.time()
        options['tmp_scan'] = scan
        # --------------------------------------------------
        # move things to a tmp folder before starting
        # --------------------------------------------------

        current_folder = os.path.join(options['test_folder'], scan)
        options['tmp_folder'] = os.path.normpath(
            os.path.join(current_folder,  'tmp'))

        # --------------------------------------------------
        # preprocess scans
        # --------------------------------------------------
        preprocess_scan(current_folder, options)

        # --------------------------------------------------
        # WM MS lesion inference
        # --------------------------------------------------
        seg_time = time.time()

        "> CNN:", scan, "running WM lesion segmentation"
        sys.stdout.flush()
        options['test_scan'] = scan

        test_x_data = {scan: {m: os.path.join(options['tmp_folder'], n)
                              for m, n in zip(options['modalities'],
                                              options['x_names'])}}

        test_cascaded_model(model, test_x_data, options)
        print "> INFO:", scan, "CNN Segmentation time: ", round(time.time() - seg_time), "sec"
        print "> INFO:", scan, "total pipeline time: ", round(time.time() - total_time), "sec"

        # remove tmps if not set
        if options['save_tmp'] is False:
            try:
                os.rmdir(options['tmp_folder'])
                os.rmdir(os.path.join(options['current_folder'],
                                      options['experiment']))
            except:
                pass

    print "> INFO: All processes have been finished. Have a good day!"
