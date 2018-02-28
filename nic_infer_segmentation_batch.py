# ------------------------------------------------------------------------------------------------------------
#   MS lesion segmentation pipeline
# ---------------------------------
#   - incorporates:
#         - MRI identification
#         - registration
#         - skull stripping
#         - MS lesion segmentation using the CNN Valverde et al (NI2017)
#
#  Sergi Valverde 2017
#  svalverde@eia.udg.edu
# ------------------------------------------------------------------------------------------------------------

import os
import argparse
import sys
import platform
import time
import ConfigParser
from utils.load_options import load_options
from utils.preprocess import preprocess_scan
from shutil import copyfile

os.system('cls' if platform.system() == 'Windows' else 'clear')
print "##################################################"
print "# MS WM lesion segmentation                      #"
print "#                                                #"
print "# -------------------------------                #"
print "# (c) Sergi Valverde 2017                        #"
print "# Neuroimage Computing Group                     #"
print "# -------------------------------                #"
print "##################################################\n"

# link related libraries
CURRENT_PATH = os.getcwd()
sys.path.append(os.path.join(CURRENT_PATH, 'libs'))

# load options from input
parser = argparse.ArgumentParser()
parser.add_argument('--docker',
                    dest='docker',
                    action='store_true')
parser.set_defaults(docker=False)
args = parser.parse_args()
container = args.docker

# --------------------------------------------------
# load default options and update them with user information
# from utils.load_options import *
# --------------------------------------------------
default_config = ConfigParser.SafeConfigParser()
default_config.read(os.path.join(CURRENT_PATH, 'config', 'default.cfg'))
user_config = ConfigParser.RawConfigParser()
user_config.read(os.path.join(CURRENT_PATH, 'config', 'configuration.cfg'))

# read user's configuration file
options = load_options(default_config, user_config)

# set GPU mode from the configuration file. So, this has to be updated before calling
# the CNN libraries if the default config "~/.theanorc" has to be replaced.
if options['mode'].find('cuda') == -1 and options['mode'].find('gpu') == -1:
    os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cpu,floatX=float32,optimizer=fast_compile'
else:
    os.environ['THEANO_FLAGS']='mode=FAST_RUN,device='+options['mode'] +',floatX=float32,optimizer=fast_compile'

from CNN.base import test_cascaded_model
from CNN.build_model_nolearn import cascade_model


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
     "> ERROR: The OS system", host_os, "is not currently supported"


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

if container:
    options['test_folder'] = os.path.normpath('/data' + options['test_folder'])
else:
    options['test_folder'] = os.path.normpath(options['test_folder'])

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

    out_seg = test_cascaded_model(model, test_x_data, options)
    print "> INFO:", scan, "CNN Segmentation time: ", round(time.time() - seg_time), "sec"

    print "> INFO:", scan, "total pipeline time: ", round(time.time() - total_time), "sec"

    # remove tmps if not set
    if options['save_tmp'] is False:
        try:
            copyfile(os.path.join(current_folder,
                                  options['experiment'],
                                  options['experiment'] +
                                  '_out_CNN.nii.gz'),
                     os.path.join(current_folder,
                                  'out_seg_' +
                                  options['experiment'] +
                                  '.nii.gz'))
            os.rmdir(options['tmp_folder'])
            os.rmdir(os.path.join(options['current_folder'],
                                  options['experiment']))
        except:
            pass

print "> INFO: All processes have been finished. Have a good day!"
