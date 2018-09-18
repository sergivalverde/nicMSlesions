# --------------------------------------------------
# load options for CNN lesion segmentation
#
# Options are loaded from a configuration file
#
# --------------------------------------------------


def load_options(default_config, user_config):
    """
    map options from user input into the default config
    """
    sections = user_config.sections()

    for s in sections:
        options = user_config.options(s)
        for o in options:
            default_config.set(s, o, user_config.get(s, o))

    # --------------------------------------------------
    # options
    # --------------------------------------------------
    options = {}

    # experiment name (where trained weights are)
    options['experiment'] = default_config.get('model', 'name')
    options['train_folder'] = default_config.get('database', 'train_folder')
    options['test_folder'] = default_config.get('database', 'inference_folder')
    options['output_folder'] = '/output'
    options['current_scan'] = 'scan'
    # options['t1_name'] = default_config.get('database', 't1_name')
    # options['flair_name'] = default_config.get('database', 'flair_name')
    options['flair_tags'] = [el.strip() for el in
                             default_config.get('database',
                                                'flair_tags').split(',')]
    options['t1_tags'] = [el.strip() for el in
                          default_config.get('database',
                                             't1_tags').split(',')]
    options['mod3_tags'] = [el.strip() for el in
                            default_config.get('database',
                                               'mod3_tags').split(',')]
    options['mod4_tags'] = [el.strip() for el in
                            default_config.get('database',
                                               'mod4_tags').split(',')]
    options['roi_tags'] = [el.strip() for el in
                           default_config.get('database',
                                              'roi_tags').split(',')]
    # options['ROI_name'] = default_config.get('database', 'ROI_name')
    options['debug'] = default_config.get('database', 'debug')

    modalities = [str(options['flair_tags'][0]),
                  options['t1_tags'][0],
                  options['mod3_tags'][0],
                  options['mod4_tags'][0]]
    names = ['FLAIR', 'T1', 'MOD3', 'MOD4']

    options['modalities'] = [n for n, m in
                             zip(names, modalities) if m != 'None']
    options['image_tags'] = [m for m in modalities if m != 'None']
    options['x_names'] = [n + '_brain.nii.gz' for n, m in
                          zip(names, modalities) if m != 'None']

    options['out_name'] = 'out_seg.nii.gz'

    # preprocessing
    options['register_modalities'] = (default_config.get('database',
                                                         'register_modalities'))
    options['denoise'] = (default_config.get('database',
                                             'denoise'))
    options['denoise_iter'] = (default_config.getint('database',
                                                     'denoise_iter'))
    options['skull_stripping'] = (default_config.get('database',
                                                     'skull_stripping'))
    options['save_tmp'] = (default_config.get('database', 'save_tmp'))

    # net options
    # options['gpu_mode'] = default_config.get('model', 'gpu_mode')
    options['gpu_number'] = default_config.getint('model', 'gpu_number')
    options['pretrained'] = default_config.get('model', 'pretrained')
    options['min_th'] = 0.5
    options['fully_convolutional'] = False
    options['patch_size'] = (11, 11, 11)
    options['weight_paths'] = None
    options['train_split'] = default_config.getfloat('model', 'train_split')
    options['max_epochs'] = default_config.getint('model', 'max_epochs')
    options['patience'] = default_config.getint('model', 'patience')
    options['batch_size'] = default_config.getint('model', 'batch_size')
    options['net_verbose'] = default_config.getint('model', 'net_verbose')
    # options['load_weights'] = default_config.get('model', 'load_weights')
    options['load_weights'] = True
    options['randomize_train'] = True

    # post processing options
    options['t_bin'] = default_config.getfloat('postprocessing', 't_bin')
    options['l_min'] = default_config.getint('postprocessing', 'l_min')
    options['min_error'] = default_config.getfloat('postprocessing',
                                                   'min_error')

    # training options
    options['full_train'] = (default_config.get('train', 'full_train'))
    options['pretrained_model'] = default_config.get('train',
                                                     'pretrained_model')

    options['balanced_training'] = default_config.get('train',
                                                      'balanced_training')

    options['fract_negative_positive'] = default_config.getfloat('train',
                                                                 'fraction_negatives')
    options['num_layers'] = None

    options = parse_values_to_types(options)
    return options


def parse_values_to_types(options):
    """
    process values into types
    """

    keys = options.keys()
    for k in keys:
        value = options[k]
        if value == 'True':
            options[k] = True
        if value == 'False':
            options[k] = False

    return options


def print_options(options):
    """
    print options
    """
    print "--------------------------------------------------"
    print " configuration options:"
    print "--------------------------------------------------"
    print " "
    keys = options.keys()
    for k in keys:
        print k, ':', options[k]
    print "--------------------------------------------------"
