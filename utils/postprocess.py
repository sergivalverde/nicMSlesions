import os
import signal
import subprocess
import time
import platform


def invert_registration(current_folder, options):
    """
    - Inverting the lesion masks from MPRAGE+192 space to the original FLAIR
    - Current folder/experiment contains the output segmentation masks

    """

    # rigid registration
    os_host = platform.system()
    if os_host == 'Windows':
        reg_transform = 'reg_transform.exe'
        reg_resample = 'reg_resample.exe'
    elif os_host == 'Linux':
        reg_transform = 'reg_transform'
        reg_resample = 'reg_resample'
    else:
        print "> ERROR: The OS system", os_host, "is not currently supported."

    reg_transform_path = os.path.join(options['niftyreg_path'], reg_transform)
    reg_resample_path = os.path.join(options['niftyreg_path'], reg_resample)

    # compute the inverse transformation
    try:
        subprocess.check_output([reg_transform_path, '-invAff',
                                 os.path.join(options['tmp_folder'],
                                              'FLAIR_transf.txt'),
                                 os.path.join(options['tmp_folder'],
                                              'inv_FLAIR_transf.txt')])
    except:
        print "> ERROR: computing the inverse transformation matrix.\
        Quitting program."
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)

    print "> POST: registering output segmentation masks back to FLAIR"

    current_experiment = os.path.join(current_folder, options['experiment'])
    list_scans = os.listdir(current_experiment)

    for file in list_scans:

        # compute the inverse transformation
        current_name = file[0:file.find('.')]
        try:
            subprocess.check_output([reg_resample_path,
                                     '-ref', os.path.join(options['tmp_folder'],
                                                          'FLAIR.nii.gz'),
                                     '-flo', os.path.join(current_experiment,
                                                          file),
                                     '-trans', os.path.join(options['tmp_folder'],
                                                            'inv_FLAIR_transf.txt'),
                                     '-res', os.path.join(current_experiment,
                                                          current_name + '_FLAIR.nii.gz'),
                                     '-inter', '0'])
        except:
            print "> ERROR: resampling ",  current_name, "Quitting program."
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)
