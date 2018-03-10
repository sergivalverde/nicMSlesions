import os
import shutil
import sys
import signal
import subprocess
import time
import platform
import nibabel as nib
import numpy as np
from medpy.filter.smoothing import anisotropic_diffusion as ans_dif


def get_mode(input_data):
    """
    Get the stastical mode
    """
    (_, idx, counts) = np.unique(input_data,
                                 return_index=True,
                                 return_counts=True)
    index = idx[np.argmax(counts)]
    mode = input_data[index]

    return mode


def parse_input_masks(current_folder, options):

    """
    identify input image masks parsing image name labels

    """

    if options['task'] == 'training':
        modalities = options['modalities'] + ['lesion']
        image_tags = options['image_tags'] + options['roi_tags']
    else:
        modalities = options['modalities']
        image_tags = options['image_tags']

    if options['debug']:
        print "> DEBUG:", "number of input sequences to find:", len(modalities)
    scan = options['tmp_scan']
    masks = os.listdir(current_folder)

    print "> PRE:", scan, "identifying input modalities"

    found_modalities = 0
    for m in masks:
        if m.find('.nii') > 0:
            input_path = os.path.join(current_folder, m)
            input_sequence = nib.load(input_path)
            # check first the input modalities
            # find tag
            found_mod = [m.find(tag) if m.find(tag) >= 0
                         else np.Inf for tag in image_tags]
            if found_mod[np.argmin(found_mod)] is not np.Inf:
                mod = modalities[np.argmin(found_mod)]
                # generate a new output image modality
                # check for extra dimensions
                input_image = np.squeeze(input_sequence.get_data())
                output_sequence = nib.Nifti1Image(input_image,
                                                  affine=input_sequence.affine)
                output_sequence.to_filename(
                    os.path.join(options['tmp_folder'], mod + '.nii.gz'))

                found_modalities += 1

                if options['debug']:
                    print "    --> ", m, "as", mod, "image"

    # check that the minimum number of modalities are used
    if found_modalities < len(modalities):
        print "> ERROR:", scan, \
            "does not contain all valid input modalities"
        sys.stdout.flush()
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)


def register_masks(options):
    """
    - to doc
    - moving all images to the MPRAGE+192 space

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

    for mod in options['modalities']:
        if mod == 'T1':
            continue

        try:
            print "> PRE:", scan, "registering",  mod, " --> T1 space"
            subprocess.check_output([reg_aladin_path, '-ref',
                                     os.path.join(options['tmp_folder'], 'T1.nii.gz'),
                                     '-flo', os.path.join(options['tmp_folder'], mod + '.nii.gz'),
                                     '-aff', os.path.join(options['tmp_folder'], mod + '_transf.txt'),
                                     '-res', os.path.join(options['tmp_folder'], 'r' + mod + '.nii.gz')])
        except:
            print "> ERROR:", scan, "registering masks on  ", mod, "quiting program."
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)

    # if training, the lesion mask is also registered through the T1 space.
    # Assuming that the refefence lesion space was FLAIR.
    if options['task'] == 'training':
        # rigid registration
        os_host = platform.system()
        if os_host == 'Windows':
            reg_exe = 'reg_resample.exe'
        elif os_host == 'Linux':
            reg_exe = 'reg_resample'
        else:
            print "> ERROR: The OS system", os_host, "is not currently supported."

        reg_resample_path = os.path.join(options['niftyreg_path'], reg_exe)

        try:
            print "> PRE:", scan, "resampling the lesion mask --> T1 space"
            subprocess.check_output([reg_resample_path, '-ref',
                                     os.path.join(options['tmp_folder'], 'T1.nii.gz'),
                                     '-flo', os.path.join(options['tmp_folder'], 'lesion'),
                                     '-trans', os.path.join(options['tmp_folder'], 'FLAIR_transf.txt'),
                                     '-res', os.path.join(options['tmp_folder'], 'lesion.nii.gz'),
                                     '-inter', '0'])
        except:
            print "> ERROR:", scan, "registering masks on  ", mod, "quiting program."
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)


def denoise_masks(options):
    """
    Denoise input masks to reduce noise.
    Using anisotropic Diffusion (Perona and Malik)

    """

    for mod in options['modalities']:

        current_image = mod + '.nii.gz' if mod == 'T1'\
                        else 'r' + mod + '.nii.gz'

        tmp_scan = nib.load(os.path.join(options['tmp_folder'],
                                         current_image))

        tmp_scan.get_data()[:] = ans_dif(tmp_scan.get_data(),
                                         niter=options['denoise_iter'])

        tmp_scan.to_filename(os.path.join(options['tmp_folder'],
                                          'd' + current_image))
        if options['debug']:
            print "> DEBUG: Denoising ", current_image


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
    t1_im = os.path.join(options['tmp_folder'], 'dT1.nii.gz')
    t1_st_im = os.path.join(options['tmp_folder'], 'T1_brain.nii.gz')

    try:
        print "> PRE:", scan, "skull_stripping the T1 modality"
        subprocess.check_output([options['robex_path'],
                                 t1_im,
                                 t1_st_im])
    except:
        print "> ERROR:", scan, "registering masks, quiting program."
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)

    brainmask = nib.load(t1_st_im).get_data() > 1
    for mod in options['modalities']:

        if mod == 'T1':
            continue

        # apply the same mask to the rest of modalities to reduce
        # computational time

        print '> PRE: ', scan, 'Applying skull mask to ', mod, 'image'
        current_mask = os.path.join(options['tmp_folder'],
                                    'dr' + mod + '.nii.gz')
        current_st_mask = os.path.join(options['tmp_folder'],
                                       mod + '_brain.nii.gz')

        mask = nib.load(current_mask)
        mask_nii = mask.get_data()
        mask_nii[brainmask == 0] = 0
        mask.get_data()[:] = mask_nii
        mask.to_filename(current_st_mask)


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
    id_time = time.time()
    parse_input_masks(current_folder, options)
    print "> INFO:", scan, "elapsed time: ", round(time.time() - id_time), "sec"

    # --------------------------------------------------
    # register modalities
    # --------------------------------------------------
    if options['register_modalities'] is True:
        reg_time = time.time()
        register_masks(options)
        print "> INFO:", scan, "elapsed time: ", round(time.time() - reg_time), "sec"
    else:
        try:
            for mod in options['modalities']:
                if mod == 'T1':
                    continue
                out_scan = mod + '.nii.gz' if mod == 'T1' else 'r' + mod + '.nii.gz'
                shutil.copy2(os.path.join(options['tmp_folder'],
                                         mod + '.nii.gz'),
                             os.path.join(options['tmp_folder'],
                                         out_scan))
        except:
            print "> ERROR: registration ", scan, "I can not rename input modalities as tmp files. Quiting program."

            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)

    # --------------------------------------------------
    # noise filtering
    # --------------------------------------------------
    if options['denoise'] is True:
        denoise_time = time.time()
        denoise_masks(options)
        print "> INFO: denoising", scan, "elapsed time: ", round(time.time() - denoise_time), "sec"
    else:
        try:
            for mod in options['modalities']:
                input_scan = mod + '.nii.gz' if mod == 'T1' else 'r' + mod + '.nii.gz'
                shutil.copy(os.path.join(options['tmp_folder'],
                                         input_scan),
                            os.path.join(options['tmp_folder'],
                                         'd' + input_scan))
        except:
            print "> ERROR denoising:", scan, "I can not rename input modalities as tmp files. Quiting program."
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
            for mod in options['modalities']:
                input_scan = 'd' + mod + '.nii.gz' if mod == 'T1' else 'dr' + mod + '.nii.gz'
                shutil.copy(os.path.join(options['tmp_folder'],
                                         input_scan),
                            os.path.join(options['tmp_folder'],
                                         mod + '_brain.nii.gz'))
        except:
            print "> ERROR: Skull-stripping", scan, "I can not rename input modalities as tmp files. Quiting program."
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)

    if options['skull_stripping'] is True and options['register_modalities'] is True:
        print "> INFO:", scan, "total preprocessing time: ", round(time.time() - preprocess_time)
