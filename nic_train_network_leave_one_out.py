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
import argparse
import ConfigParser
from utils.preprocess import preprocess_scan
from utils.load_options import load_options

os.system('cls' if platform.system() == 'Windows' else 'clear')

print "##################################################"
print "# ------------                                   #"
print "# nicMSlesions                                   #"
print "# ------------                                   #"
print "# MS WM lesion segmentation                      #"
print "#                                                #"
print "# -------------------------------                #"
print "# (c) Sergi Valverde 2017                        #"
print "# Neuroimage Computing Group                     #"
print "# -------------------------------                #"
print "##################################################\n"


# load options from input
parser = argparse.ArgumentParser()
parser.add_argument('--docker',
                    dest='docker',
                    action='store_true')
parser.set_defaults(docker=False)
args = parser.parse_args()
container = args.docker

# link related libraries
CURRENT_PATH = os.getcwd()
sys.path.append(os.path.join(CURRENT_PATH, 'libs'))

# load default options and update them with user information
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
        os.path.join(CURRENT_PATH, 'libs', 'win', 'niftyreg'))
    options['robex_path'] = os.path.normpath(
        os.path.join(CURRENT_PATH, 'libs', 'win', 'ROBEX', 'runROBEX.bat'))
    options['test_slices'] = 256
else:
    print "The OS system", host_os, "is not currently supported."
    exit()


# set GPU mode from the configuration file. This has to be updated
# calling the CNN libraries if the default config "~/.theanorc" has to be
# replaced.
if options['mode'].find('gpu') == -1 and options['mode'].find('cuda') == -1:
    os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cpu,floatX=float32,optimizer=fast_compile'
else:
    os.environ['THEANO_FLAGS']='mode=FAST_RUN,device='+options['mode'] +',floatX=float32,optimizer=fast_compile'

from CNN.base import train_cascaded_model, test_cascaded_model
from CNN.build_model_nolearn import cascade_model

if container:
    options['train_folder'] = os.path.normpath(
        '/data' + options['train_folder'])
else:
    options['train_folder'] = os.path.normpath(options['train_folder'])

scan_list = os.listdir(options['train_folder'])
scan_list.sort()

# --------------------------------------------------
# process all scans before leave-one-out-training
# move things to a tmp folder before starting
# --------------------------------------------------
for scan in scan_list:

    total_time = time.time()
    preprocess_time = time.time()

    options['tmp_scan'] = scan
    current_folder = os.path.join(options['train_folder'], scan)
    options['tmp_folder'] = os.path.normpath(
        os.path.join(current_folder,  'tmp'))

    # preprocess scan
    preprocess_scan(current_folder, options)


# --------------------------------------------------
# WM MS lesion training
# - configure net and train leave one out
# --------------------------------------------------

for scan in scan_list:
    seg_time = time.time()
    print "> CNN: Starting training session for scan", scan

    # select training scans

    train_x_data = {f: {m: os.path.join(options['train_folder'], f, 'tmp', n)
                        for m, n in zip(options['modalities'],
                                        options['x_names'])}
                    for f in scan_list if f != scan}
    train_y_data = {f: os.path.join(options['train_folder'], f, 'tmp',
                                    options['ROI_name'])
                    for f in scan_list if f != scan}

    # organize the experiment: save models and traiining images inside a
    # predifined folder. Network parameters and weights are stored inside
    # test_folder/experiment/nets/
    # training images are stored inside test_folder/experiment/.train
    # final segmentation images are stored in test_folder/experiment

    if not os.path.exists(os.path.join(options['train_folder'],
                                       scan,
                                       'nets')):
        os.mkdir(os.path.join(options['train_folder'], scan, 'nets'))

    options['weight_paths'] = os.path.join(options['train_folder'],
                                           scan,
                                           'nets')
    options['load_weights'] = False
    options['test_scan'] = scan

    # train the model for the current scan
    print "> CNN: training net with %d subjects" %(len(train_x_data.keys()))

    # --------------------------------------------------
    # initialize the CNN and train the classifier
    # --------------------------------------------------
    model = cascade_model(options)
    model = train_cascaded_model(model, train_x_data, train_y_data,  options)

    print "> INFO: training time:", round(time.time() - seg_time), "sec"
    print "> INFO: total pipeline time: ", round(time.time() - total_time), "sec"

    # --------------------------------------------------
    # test the current scan
    # --------------------------------------------------
    test_x_data = {scan: {m: os.path.join(options['train_folder'], scan, n)
                          for m, n in zip(options['modalities'],
                                          options['x_names'])}}

    out_seg = test_cascaded_model(model, test_x_data, options)


print "> INFO: All processes have been finished. Have a good day!"
