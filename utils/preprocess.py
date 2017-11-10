import os
import shutil
import sys
import signal
import subprocess
import time
import platform
import nibabel as nib
import numpy as np

def get_mode(input_data):
    """
    """
    (_, idx, counts) = np.unique(input_data,
                                 return_index=True,
                                 return_counts=True)
    index = idx[np.argmax(counts)]
    mode = input_data[index]

    return mode


def parse_input_masks(current_folder, options):

    """
    identify t1-w and FLAIR masks parsing image name labels

    """

    flair_tags = options['flair_tags']
    t1_tags = options['t1_tags']
    roi_tags = options['roi_tags']
    f_s, t1_s, r_s = False, False, False

    scan = options['tmp_scan']    
    masks = os.listdir(current_folder)
    print "> PRE:", scan, "identifying input modalities"
    for m in masks:
        if m.find('.nii') > 0:

            input_path = os.path.join(current_folder, m)
            input_sequence = nib.load(input_path)

            # find tags
            f_search = len([m.find(tag) for
                            tag in flair_tags if m.find(tag) >= 0]) > 0
            t1_search = len([m.find(tag) for
                             tag in t1_tags if m.find(tag) >= 0]) > 0
            r_search = len([m.find(tag) for
                            tag in roi_tags if m.find(tag) >= 0]) > 0

            # roi
            if r_search is True:
                # r_s = True
                input_sequence.to_filename(
                    os.path.join(options['tmp_folder'], 'lesion.nii.gz'))
                if options['debug']:
                    print "    --> ", m, "as ROI image"
            elif f_search is True:
                f_s = True
                input_sequence.to_filename(
                    os.path.join(options['tmp_folder'], 'FLAIR.nii.gz'))
                if options['debug']:
                    print "    --> ", m, "as FLAIR image"
            elif t1_search is True:
                t1_s = True
                input_sequence.to_filename(
                    os.path.join(options['tmp_folder'], 'T1.nii.gz'))
                if options['debug']:
                    print "    --> ", m, "as T1 image"
            else:
                if options['debug']:
                    print "    --> ", m, "not identified"
                pass

    # check that the minimum number of modalities are used
    if f_s is False or t1_s is False:
        print "> ERROR:", scan, \
            "does not contain all valid input modalities"
        if f_search is False:
            print "> ERROR:", scan, "FLAIR modality not found"
        if t1_search is False: 
            print "> ERROR:", scan, "T1 modality not found"
        sys.stdout.flush()
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)


def register_masks(options):
    """
    - to doc
    - moving T1-w images to FLAIR space

    """

    scan = options['tmp_scan']
    # rigid registration
    os_host = platform.system()
    if os_host == 'Windows':
        reg_exe = 'reg_aladin.exe'
    elif os_host == 'Linux':
        reg_exe = 'reg_aladin'
    else:
        print "> ERROR: The OS system", os_host, "is not currently supported."

    reg_aladin_path = os.path.join(options['niftyreg_path'], reg_exe)
    try:
        print "> PRE:", scan, "registering T1-w --> FLAIR"
        subprocess.check_output([reg_aladin_path, '-ref',
                        os.path.join(options['tmp_folder'], 'FLAIR.nii.gz'),
                        '-flo' , os.path.join(options['tmp_folder'], 'T1.nii.gz'),
                        '-aff' , os.path.join(options['tmp_folder'], 'transf.txt'),
                        '-res' , os.path.join(options['tmp_folder'], 'rT1.nii.gz')])
    except:
        print "> ERROR:", scan, "registering masks, quiting program."
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)


def skull_strip(options):
    """
    External skull stripping using ROBEX: Run Robex and save skull
    stripped masks
    input:
       - options: contains the path to input images
    output:
    - None
    """
    scan = options['tmp_scan']
    flair_im = os.path.join(options['tmp_folder'],
                                             'FLAIR.nii.gz')
    t1_im = os.path.join(options['tmp_folder'],
                                          'rT1.nii.gz')
    flair_st_im = os.path.join(options['tmp_folder'],
                                                'FLAIR_brain.nii.gz')
    t1_st_im = os.path.join(options['tmp_folder'],
                                             'T1_brain.nii.gz')

    try:
        print "> PRE:", scan, "skull_stripping input modalities"
        subprocess.check_output([options['robex_path'],
                                 flair_im,
                                 flair_st_im])
    except:
        print "> ERROR:", scan, "registering masks, quiting program."
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)

    # apply the same mask to T1-w to reduce computational time

    flair_mask = nib.load(flair_st_im)
    t1_mask = nib.load(t1_im)
    T1 = t1_mask.get_data()
    T1[flair_mask.get_data() < 1] = 0
    t1_mask.get_data()[:] = T1
    t1_mask.to_filename(t1_st_im)


def preprocess_scan(current_folder, options):
    """
    Preprocess scan taking into account user options
    - input:
      current_folder = path to the current image
      options: options

    """
    preprocess_time = time.time()

    scan = options['tmp_scan']
    try:
        # os.rmdir(os.path.join(current_folder,  'tmp'))
        os.mkdir(options['tmp_folder'])
    except:
        if os.path.exists(options['tmp_folder']) is False:
            print "> ERROR:",  scan, "I can not create tmp folder for", current_folder, "Quiting program."
                
        else:
            pass

    # --------------------------------------------------
    # find modalities
    # --------------------------------------------------

    if options['t1_name'] == 'None':
        id_time = time.time()
        parse_input_masks(current_folder, options)
        print "> INFO:", scan, "elapsed time: ", round(time.time() - id_time), "sec"
    else:
        try:
            shutil.move(os.path.join(current_folder, options['t1_name']),
                      os.path.join(options['tmp_folder'], 'T1.nii.gz'))
            shutil.move(os.path.join(current_folder, options['flair_name']),
                      os.path.join(options['tmp_folder'], 'FLAIR.nii.gz'))
        except:
            print "> ERROR:", scan, options['t1_name'], \
                'or', options['flair_name'], "do not appear to exist..\
                Quiting program"
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)

    # --------------------------------------------------
    # register modalities
    # --------------------------------------------------

    if options['register_modalities'] is True:
        reg_time = time.time()
        register_masks(options)
        print "> INFO:", scan, "elapsed time: ", round(time.time() - reg_time), "sec"
    else:
        try:
            shutil.move(os.path.join(options['tmp_folder'], 'T1.nii.gz'),
                      os.path.join(options['tmp_folder'], 'rT1.nii.gz'))
        except:
            print "> ERROR:", scan, "I can not rename input modalities as tmp files. Quiting program."
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)

    # --------------------------------------------------
    # skull strip
    # --------------------------------------------------

    if options['skull_stripping'] is True:
        sk_time = time.time()
        skull_strip(options)
        print "> INFO:", scan, "elapsed time: ", round(time.time() - sk_time), "sec"
    else:
        try:
            shutil.move(os.path.join(options['tmp_folder'], 'rT1.nii.gz'),
                      os.path.join(options['tmp_folder'], 'T1_brain.nii.gz'))
            shutil.move(os.path.join(options['tmp_folder'], 'FLAIR.nii.gz'),
                      os.path.join(options['tmp_folder'], 'FLAIR_brain.nii.gz'))
        except:
            print "> ERROR:", scan, "I can not rename input modalities as tmp files. Quiting program."
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)

    if options['skull_stripping'] is True and options['register_modalities'] is True:
        print "> INFO:", scan, "total preprocessing time: ", round(time.time() - preprocess_time)
        
