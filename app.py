# --------------------------------------------------
# nicMSlesions: MS white matter lesion segmentation
#
# Copyright Sergi Valverde 2017
# Neuroimage Computing Group
# http://atc.udg.edu/nic/about.html
#
# Licensed under the BSD 2-Clause license. A copy of
# the license is present in the root directory.
#
# --------------------------------------------------

import ConfigParser
import argparse
import platform
import subprocess
import os
import signal
import Queue
import threading
from __init__ import __version__
from Tkinter import Frame, LabelFrame, Label, END, Tk
from Tkinter import Entry, Button, Checkbutton, OptionMenu, Toplevel
from Tkinter import BooleanVar, StringVar, IntVar, DoubleVar
from tkFileDialog import askdirectory
from ttk import Notebook
from PIL import Image, ImageTk
import webbrowser
from cnn_scripts import train_network, infer_segmentation, get_config


class wm_seg:
    """
    Simple GUI application
    If the application inside a container, automatic updates are removed.

    The application uses two frames (tabs):
    - training
    - testing
    """
    def __init__(self, master, container):

        self.master = master
        master.title("nicMSlesions")

        # running on a container
        self.container = container

        # gui attributes
        self.path = os.getcwd()
        self.default_config = None
        self.user_config = None
        self.current_folder = os.getcwd()
        self.list_train_pretrained_nets = []
        self.list_test_nets = []
        self.version = __version__
        if self.container is False:
            # version_number
            self.commit_version = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'])

        # queue and thread parameters. All processes are embedded
        # inside threads to avoid freezing the application
        self.train_task = None
        self.test_task = None
        self.test_queue = Queue.Queue()
        self.train_queue = Queue.Queue()

        # --------------------------------------------------
        # parameters. Mostly from the config/*.cfg files
        # --------------------------------------------------

        # data parameters
        self.param_training_folder = StringVar()
        self.param_test_folder = StringVar()
        self.param_FLAIR_tag = StringVar()
        self.param_T1_tag = StringVar()
        self.param_MOD3_tag = StringVar()
        self.param_MOD4_tag = StringVar()
        self.param_mask_tag = StringVar()
        self.param_model_tag = StringVar()
        self.param_register_modalities = BooleanVar()
        self.param_skull_stripping = BooleanVar()
        self.param_denoise = BooleanVar()
        self.param_denoise_iter = IntVar()
        self.param_save_tmp = BooleanVar()
        self.param_debug = BooleanVar()

        # train parameters
        self.param_net_folder = os.path.join(self.current_folder, 'nets')
        self.param_use_pretrained_model = BooleanVar()
        self.param_pretrained_model = StringVar()
        self.param_inference_model = StringVar()
        self.param_num_layers = IntVar()
        self.param_net_name = StringVar()
        self.param_net_name.set('None')
        self.param_balanced_dataset = BooleanVar()
        self.param_fract_negatives = DoubleVar()

        # model parameters
        self.param_pretrained = None
        self.param_min_th = DoubleVar()
        self.param_patch_size = IntVar()
        self.param_weight_paths = StringVar()
        self.param_load_weights = BooleanVar()
        self.param_train_split = DoubleVar()
        self.param_max_epochs = IntVar()
        self.param_patience = IntVar()
        self.param_batch_size = IntVar()
        self.param_net_verbose = IntVar()
        self.param_t_bin = DoubleVar()
        self.param_l_min = IntVar()
        self.param_min_error = DoubleVar()
        self.param_mode = BooleanVar()
        self.param_gpu_number = IntVar()

        # load the default configuration from the conf file
        self.load_default_configuration()

        # self frame (tabbed notebook)
        self.note = Notebook(self.master)
        self.note.pack()

        os.system('cls' if platform.system() == 'Windows' else 'clear')
        print "##################################################"
        print "# ------------                                   #"
        print "# nicMSlesions                                   #"
        print "# ------------                                   #"
        print "# MS WM lesion segmentation                      #"
        print "#                                                #"
        print "# -------------------------------                #"
        print "# (c) Sergi Valverde 2019                        #"
        print "# Neuroimage Computing Group                     #"
        print "# -------------------------------                #"
        print "##################################################\n"
        print "Please select options for training or inference in the menu..."

        # --------------------------------------------------
        # training tab
        # --------------------------------------------------
        self.train_frame = Frame()
        self.note.add(self.train_frame, text="Training")
        self.test_frame = Frame()
        self.note.add(self.test_frame, text="Inference")

        # label frames
        cl_s = 5
        self.tr_frame = LabelFrame(self.train_frame, text="Training images:")
        self.tr_frame.grid(row=0, columnspan=cl_s, sticky='WE',
                           padx=5, pady=5, ipadx=5, ipady=5)
        self.model_frame = LabelFrame(self.train_frame, text="CNN model:")
        self.model_frame.grid(row=5, columnspan=cl_s, sticky='WE',
                              padx=5, pady=5, ipadx=5, ipady=5)

        # training options
        self.inFolderLbl = Label(self.tr_frame, text="Training folder:")
        self.inFolderLbl.grid(row=0, column=0, sticky='E', padx=5, pady=2)
        self.inFolderTxt = Entry(self.tr_frame)
        self.inFolderTxt.grid(row=0,
                              column=1,
                              columnspan=5,
                              sticky="W",
                              pady=3)
        self.inFileBtn = Button(self.tr_frame, text="Browse ...",
                                command=self.load_training_path)
        self.inFileBtn.grid(row=0,
                            column=5,
                            columnspan=1,
                            sticky='W',
                            padx=5,
                            pady=1)

        self.optionsBtn = Button(self.tr_frame,
                                 text="Other options",
                                 command=self.parameter_window)
        self.optionsBtn.grid(row=0,
                             column=10,
                             columnspan=1,
                             sticky="W",
                             padx=(100, 1),
                             pady=1)


        # setting input modalities: FLAIR + T1 are mandatory
        # Mod 3 / 4 are optional
        self.flairTagLbl = Label(self.tr_frame, text="FLAIR tag:")
        self.flairTagLbl.grid(row=1, column=0, sticky='E', padx=5, pady=2)
        self.flairTxt = Entry(self.tr_frame,
                              textvariable=self.param_FLAIR_tag)
        self.flairTxt.grid(row=1, column=1, columnspan=1, sticky="W", pady=1)

        self.t1TagLbl = Label(self.tr_frame, text="T1 tag:")
        self.t1TagLbl.grid(row=2, column=0, sticky='E', padx=5, pady=2)
        self.t1Txt = Entry(self.tr_frame, textvariable=self.param_T1_tag)
        self.t1Txt.grid(row=2, column=1, columnspan=1, sticky="W", pady=1)

        self.mod3TagLbl = Label(self.tr_frame, text="mod 3 tag:")
        self.mod3TagLbl.grid(row=3, column=0, sticky='E', padx=5, pady=2)
        self.mod3Txt = Entry(self.tr_frame,
                              textvariable=self.param_MOD3_tag)
        self.mod3Txt.grid(row=3, column=1, columnspan=1, sticky="W", pady=1)

        self.mod4TagLbl = Label(self.tr_frame, text="mod 4 tag:")
        self.mod4TagLbl.grid(row=4, column=0, sticky='E', padx=5, pady=2)
        self.mod4Txt = Entry(self.tr_frame,
                              textvariable=self.param_MOD4_tag)
        self.mod4Txt.grid(row=4, column=1, columnspan=1, sticky="W", pady=1)

        self.maskTagLbl = Label(self.tr_frame, text="MASK tag:")
        self.maskTagLbl.grid(row=5, column=0,
                             sticky='E', padx=5, pady=2)
        self.maskTxt = Entry(self.tr_frame, textvariable=self.param_mask_tag)
        self.maskTxt.grid(row=5, column=1, columnspan=1, sticky="W", pady=1)

        # model options
        self.modelTagLbl = Label(self.model_frame, text="Model name:")
        self.modelTagLbl.grid(row=6, column=0,
                              sticky='E', padx=5, pady=2)
        self.modelTxt = Entry(self.model_frame,
                              textvariable=self.param_net_name)
        self.modelTxt.grid(row=6, column=1, columnspan=1, sticky="W", pady=1)

        self.checkPretrain = Checkbutton(self. model_frame,
                                         text="use pretrained",
                                         var=self.param_use_pretrained_model)
        self.checkPretrain.grid(row=6, column=3, padx=5, pady=5)

        self.update_pretrained_nets()

        self.pretrainTxt = OptionMenu(self.model_frame,
                                      self.param_pretrained_model,
                                      *self.list_train_pretrained_nets)
        self.pretrainTxt.grid(row=6, column=5, sticky='E', padx=5, pady=5)

        # START button links
        self.trainingBtn = Button(self.train_frame,
                                  state='disabled',
                                  text="Start training",
                                  command=self.train_net)
        self.trainingBtn.grid(row=7, column=0, sticky='W', padx=1, pady=1)

        # --------------------------------------------------
        # inference tab
        # --------------------------------------------------
        self.tt_frame = LabelFrame(self.test_frame, text="Inference images:")
        self.tt_frame.grid(row=0, columnspan=cl_s, sticky='WE',
                           padx=5, pady=5, ipadx=5, ipady=5)
        self.test_model_frame = LabelFrame(self.test_frame, text="CNN model:")
        self.test_model_frame.grid(row=5, columnspan=cl_s, sticky='WE',
                                   padx=5, pady=5, ipadx=5, ipady=5)

        # testing options
        self.test_inFolderLbl = Label(self.tt_frame, text="Testing folder:")
        self.test_inFolderLbl.grid(row=0, column=0, sticky='E', padx=5, pady=2)
        self.test_inFolderTxt = Entry(self.tt_frame)
        self.test_inFolderTxt.grid(row=0,
                                   column=1,
                                   columnspan=5,
                                   sticky="W",
                                   pady=3)
        self.test_inFileBtn = Button(self.tt_frame, text="Browse ...",
                                     command=self.load_testing_path)
        self.test_inFileBtn.grid(row=0,
                                 column=5,
                                 columnspan=1,
                                 sticky='W',
                                 padx=5,
                                 pady=1)

        self.test_optionsBtn = Button(self.tt_frame,
                                      text="Other options",
                                      command=self.parameter_window)
        self.test_optionsBtn.grid(row=0,
                                  column=10,
                                  columnspan=1,
                                  sticky="W",
                                  padx=(100, 1),
                                  pady=1)

        self.test_flairTagLbl = Label(self.tt_frame, text="FLAIR tag:")
        self.test_flairTagLbl.grid(row=1, column=0, sticky='E', padx=5, pady=2)
        self.test_flairTxt = Entry(self.tt_frame,
                              textvariable=self.param_FLAIR_tag)
        self.test_flairTxt.grid(row=1, column=1, columnspan=1, sticky="W", pady=1)

        self.test_t1TagLbl = Label(self.tt_frame, text="T1 tag:")
        self.test_t1TagLbl.grid(row=2, column=0, sticky='E', padx=5, pady=2)
        self.test_t1Txt = Entry(self.tt_frame, textvariable=self.param_T1_tag)
        self.test_t1Txt.grid(row=2, column=1, columnspan=1, sticky="W", pady=1)

        self.test_mod3TagLbl = Label(self.tt_frame, text="mod 3 tag:")
        self.test_mod3TagLbl.grid(row=3, column=0, sticky='E', padx=5, pady=2)
        self.test_mod3Txt = Entry(self.tt_frame,
                              textvariable=self.param_MOD3_tag)
        self.test_mod3Txt.grid(row=3, column=1, columnspan=1, sticky="W", pady=1)

        self.test_mod4TagLbl = Label(self.tt_frame, text="mod 4 tag:")
        self.test_mod4TagLbl.grid(row=4, column=0, sticky='E', padx=5, pady=2)
        self.test_mod4Txt = Entry(self.tt_frame,
                              textvariable=self.param_MOD4_tag)
        self.test_mod4Txt.grid(row=4, column=1, columnspan=1, sticky="W", pady=1)

        self.test_pretrainTxt = OptionMenu(self.test_model_frame,
                                           self.param_inference_model,
                                           *self.list_test_nets)

        self.param_inference_model.set('None')
        self.test_pretrainTxt.grid(row=5, column=0, sticky='E', padx=5, pady=5)

        # START button links cto docker task
        self.inferenceBtn = Button(self.test_frame,
                                   state='disabled',
                                   text="Start inference",
                                   command=self.infer_segmentation)
        self.inferenceBtn.grid(row=7, column=0, sticky='W', padx=1, pady=1)

        # train / test ABOUT button
        self.train_aboutBtn = Button(self.train_frame,
                                     text="about",
                                     command=self.about_window)
        self.train_aboutBtn.grid(row=7,
                                 column=4,
                                 sticky='E',
                                 padx=(1, 1),
                                 pady=1)

        self.test_aboutBtn = Button(self.test_frame,
                                    text="about",
                                    command=self.about_window)
        self.test_aboutBtn.grid(row=7,
                                column=4,
                                sticky='E',
                                padx=(1, 1),
                                pady=1)

        # Processing state
        self.process_indicator = StringVar()
        self.process_indicator.set(' ')
        self.label_indicator = Label(master,
                                     textvariable=self.process_indicator)
        self.label_indicator.pack(side="left")

        # Closing processing events is implemented via
        # a master protocol
        self.master.protocol("WM_DELETE_WINDOW", self.close_event)

    def parameter_window(self):
        """
        Setting other parameters using an emerging window
        CNN parameters, CUDA device, post-processing....

        """
        t = Toplevel(self.master)
        t.wm_title("Other parameters")

        # data parameters
        t_data = LabelFrame(t, text="data options:")
        t_data.grid(row=0, sticky="WE")
        checkPretrain = Checkbutton(t_data,
                                    text="Register modalities",
                                    var=self.param_register_modalities)
        checkPretrain.grid(row=0, sticky='W')
        checkSkull = Checkbutton(t_data,
                                 text="Skull-strip modalities",
                                 var=self.param_skull_stripping)
        checkSkull.grid(row=1, sticky="W")
        checkDenoise = Checkbutton(t_data,
                                   text="Denoise masks",
                                   var=self.param_denoise)
        checkDenoise.grid(row=2, sticky="W")

        denoise_iter_label = Label(t_data, text=" Denoise iter:               ")
        denoise_iter_label.grid(row=3, sticky="W")
        denoise_iter_entry = Entry(t_data, textvariable=self.param_denoise_iter)
        denoise_iter_entry.grid(row=3, column=1, sticky="E")

        check_tmp = Checkbutton(t_data,
                                text="Save tmp files",
                                var=self.param_save_tmp)
        check_tmp.grid(row=4, sticky="W")
        checkdebug = Checkbutton(t_data,
                                 text="Debug mode",
                                 var=self.param_debug)
        checkdebug.grid(row=5, sticky="W")

        # model parameters
        t_model = LabelFrame(t, text="Model:")
        t_model.grid(row=5, sticky="EW")

        maxepochs_label = Label(t_model, text="Max epochs:                  ")
        maxepochs_label.grid(row=6, sticky="W")
        maxepochs_entry = Entry(t_model, textvariable=self.param_max_epochs)
        maxepochs_entry.grid(row=6, column=1, sticky="E")

        trainsplit_label = Label(t_model, text="Validation %:           ")
        trainsplit_label.grid(row=7, sticky="W")
        trainsplit_entry = Entry(t_model, textvariable=self.param_train_split)
        trainsplit_entry.grid(row=7, column=1, sticky="E")

        batchsize_label = Label(t_model, text="Test batch size:")
        batchsize_label.grid(row=8, sticky="W")
        batchsize_entry = Entry(t_model, textvariable=self.param_batch_size)
        batchsize_entry.grid(row=8, column=1, sticky="E")

        mode_label = Label(t_model, text="Verbosity:")
        mode_label.grid(row=9, sticky="W")
        mode_entry = Entry(t_model, textvariable=self.param_net_verbose)
        mode_entry.grid(row=9, column=1, sticky="E")

        #gpu_mode = Checkbutton(t_model,
        #                         text="GPU:",
        #                         var=self.param_mode)
        #gpu_mode.grid(row=10, sticky="W")

        gpu_number = Label(t_model, text="GPU number:")
        gpu_number.grid(row=10, sticky="W")
        gpu_entry = Entry(t_model, textvariable=self.param_gpu_number)
        gpu_entry.grid(row=10, column=1, sticky="W")


        # training parameters
        tr_model = LabelFrame(t, text="Training:")
        tr_model.grid(row=12, sticky="EW")

        balanced_entry = Checkbutton(tr_model,
                                     text="Balanced training",
                                     command=self.check_train,
                                     var=self.param_balanced_dataset)
        balanced_entry.grid(row=13, sticky="W")


        fraction_label = Label(tr_model,
                               text="Fraction negative/positives: ")
        fraction_label.grid(row=14, sticky="W")
        self.fraction_entry = Entry(tr_model,
                                    state='disabled',
                                    textvariable=self.param_fract_negatives)
        self.fraction_entry.grid(row=14, column=1, sticky="E")

        # postprocessing parameters
        t_post = LabelFrame(t, text="Post-processing:  ")
        t_post.grid(row=15, sticky="EW")
        t_bin_label = Label(t_post, text="Out probability th:      ")
        t_bin_label.grid(row=16, sticky="W")
        t_bin_entry = Entry(t_post, textvariable=self.param_t_bin)
        t_bin_entry.grid(row=16, column=1, sticky="E")

        l_min_label = Label(t_post, text="Min out region size:         ")
        l_min_label.grid(row=17, sticky="W")
        l_min_entry = Entry(t_post, textvariable=self.param_l_min)
        l_min_entry.grid(row=17, column=1, sticky="E")

        vol_min_label = Label(t_post, text="Min vol error (ml):   ")
        vol_min_label.grid(row=18, sticky="W")
        vol_min_entry = Entry(t_post, textvariable=self.param_min_error)
        vol_min_entry.grid(row=18, column=1, sticky="E")


    def check_train(self):
        """
        Show fraction on negative / positive only when balanced training is unset
        """
        if self.fraction_entry['state'] == 'disabled':
            self.fraction_entry['state'] = 'normal'
        else:
            self.fraction_entry['state'] = 'disabled'

    def load_default_configuration(self):
        """
        load the default configuration from /config/default.cfg
        This method assign each of the configuration parameters to
        class attributes
        """

        default_config = ConfigParser.SafeConfigParser()
        default_config.read(os.path.join(self.path, 'config', 'default.cfg'))

        # dastaset parameters
        self.param_training_folder.set(default_config.get('database',
                                                          'train_folder'))
        self.param_test_folder.set(default_config.get('database',
                                                      'inference_folder'))
        self.param_FLAIR_tag.set(default_config.get('database','flair_tags'))
        self.param_T1_tag.set(default_config.get('database','t1_tags'))
        self.param_MOD3_tag.set(default_config.get('database','mod3_tags'))
        self.param_MOD4_tag.set(default_config.get('database','mod4_tags'))
        self.param_mask_tag.set(default_config.get('database','roi_tags'))
        self.param_register_modalities.set(default_config.get('database', 'register_modalities'))
        self.param_denoise.set(default_config.get('database', 'denoise'))
        self.param_denoise_iter.set(default_config.getint('database', 'denoise_iter'))
        self.param_skull_stripping.set(default_config.get('database', 'skull_stripping'))
        self.param_save_tmp.set(default_config.get('database', 'save_tmp'))
        self.param_debug.set(default_config.get('database', 'debug'))

        # train parameters
        self.param_use_pretrained_model.set(default_config.get('train', 'full_train'))
        self.param_pretrained_model.set(default_config.get('train', 'pretrained_model'))
        self.param_inference_model.set("      ")
        self.param_balanced_dataset.set(default_config.get('train', 'balanced_training'))
        self.param_fract_negatives.set(default_config.getfloat('train', 'fraction_negatives'))

        # model parameters
        self.param_net_folder = os.path.join(self.current_folder, 'nets')
        self.param_net_name.set(default_config.get('model', 'name'))
        self.param_train_split.set(default_config.getfloat('model', 'train_split'))
        self.param_max_epochs.set(default_config.getint('model', 'max_epochs'))
        self.param_patience.set(default_config.getint('model', 'patience'))
        self.param_batch_size.set(default_config.getint('model', 'batch_size'))
        self.param_net_verbose.set(default_config.get('model', 'net_verbose'))
        self.param_gpu_number.set(default_config.getint('model', 'gpu_number'))
        # self.param_mode.set(default_config.get('model', 'gpu_mode'))

        # post-processing
        self.param_l_min.set(default_config.getint('postprocessing',
                                                   'l_min'))
        self.param_t_bin.set(default_config.getfloat('postprocessing',
                                                     't_bin'))
        self.param_min_error.set(default_config.getfloat('postprocessing',
                                                     'min_error'))

    def write_user_configuration(self):
        """
        write the configuration into config/configuration.cfg
        """
        user_config = ConfigParser.RawConfigParser()

        # dataset parameters
        user_config.add_section('database')
        user_config.set('database', 'train_folder', self.param_training_folder.get())
        user_config.set('database', 'inference_folder', self.param_test_folder.get())
        user_config.set('database', 'flair_tags', self.param_FLAIR_tag.get())
        user_config.set('database', 't1_tags', self.param_T1_tag.get())
        user_config.set('database', 'mod3_tags', self.param_MOD3_tag.get())
        user_config.set('database', 'mod4_tags', self.param_MOD4_tag.get())
        user_config.set('database', 'roi_tags', self.param_mask_tag.get())

        user_config.set('database', 'register_modalities', self.param_register_modalities.get())
        user_config.set('database', 'denoise', self.param_denoise.get())
        user_config.set('database', 'denoise_iter', self.param_denoise_iter.get())
        user_config.set('database', 'skull_stripping', self.param_skull_stripping.get())
        user_config.set('database', 'save_tmp', self.param_save_tmp.get())
        user_config.set('database', 'debug', self.param_debug.get())

        # train parameters
        user_config.add_section('train')
        user_config.set('train',
                        'full_train',
                        not(self.param_use_pretrained_model.get()))
        user_config.set('train',
                        'pretrained_model',
                        self.param_pretrained_model.get())
        user_config.set('train',
                        'balanced_training',
                        self.param_balanced_dataset.get())
        user_config.set('train',
                        'fraction_negatives',
                        self.param_fract_negatives.get())
        # model parameters
        user_config.add_section('model')
        user_config.set('model', 'name', self.param_net_name.get())
        user_config.set('model', 'pretrained', self.param_pretrained)
        user_config.set('model', 'train_split', self.param_train_split.get())
        user_config.set('model', 'max_epochs', self.param_max_epochs.get())
        user_config.set('model', 'patience', self.param_patience.get())
        user_config.set('model', 'batch_size', self.param_batch_size.get())
        user_config.set('model', 'net_verbose', self.param_net_verbose.get())
        # user_config.set('model', 'gpu_mode', self.param_mode.get())
        user_config.set('model', 'gpu_number', self.param_gpu_number.get())

        # postprocessing parameters
        user_config.add_section('postprocessing')
        user_config.set('postprocessing', 't_bin', self.param_t_bin.get())
        user_config.set('postprocessing', 'l_min', self.param_l_min.get())
        user_config.set('postprocessing',
                        'min_error', self.param_min_error.get())

        # Writing our configuration file to 'example.cfg'
        with open(os.path.join(self.path,
                               'config',
                               'configuration.cfg'), 'wb') as configfile:
            user_config.write(configfile)

    def load_training_path(self):
        """
        Select training path from disk and write it.
        If the app is run inside a container,
        link the iniitaldir with /data
        """
        initialdir = '/data' if self.container else os.getcwd()
        fname = askdirectory(initialdir=initialdir)
        if fname:
            try:
                self.param_training_folder.set(fname)
                self.inFolderTxt.delete(0, END)
                self.inFolderTxt.insert(0, self.param_training_folder.get())
                self.trainingBtn['state'] = 'normal'
            except:
                pass

    def load_testing_path(self):
        """
        Selecet the inference path from disk and write it
        If the app is run inside a container,
        link the iniitaldir with /data
        """
        initialdir = '/data' if self.container else os.getcwd()
        fname = askdirectory(initialdir=initialdir)
        if fname:
            try:
                self.param_test_folder.set(fname)
                self.test_inFolderTxt.delete(0, END)
                self.test_inFolderTxt.insert(0, self.param_test_folder.get())
                self.inferenceBtn['state'] = 'normal'
            except:
                pass

    def update_pretrained_nets(self):
        """
        get a list of the  different net configuration present in the system.
        Each model configuration is represented by a folder containing the network
        weights for each of the networks. The baseline net config is always
        included by default

        """
        folders = os.listdir(self.param_net_folder)
        self.list_train_pretrained_nets = folders
        self.list_test_nets = folders

    def write_to_console(self, txt):
        """
        to doc:
        important method
        """
        self.command_out.insert(END, str(txt))

    def write_to_test_console(self, txt):
        """
        to doc:
        important method
        """
        self.test_command_out.insert(END, str(txt))

    def infer_segmentation(self):
        """
        Method implementing the inference process:
        - Check network selection
        - write the configuration to disk
        - Run the process on a new thread
        """

        if self.param_inference_model.get() == 'None':
            print "ERROR: Please, select a network model before starting...\n"
            return
        if self.test_task is None:
            self.inferenceBtn.config(state='disabled')
            self.param_net_name.set(self.param_inference_model.get())
            self.param_use_pretrained_model.set(False)
            self.write_user_configuration()
            print "\n-----------------------"
            print "Running configuration:"
            print "-----------------------"
            print "Inference model:", self.param_model_tag.get()
            print "Inference folder:", self.param_test_folder.get(), "\n"

            print "Method info:"
            print "------------"
            self.test_task = ThreadedTask(self.write_to_test_console,
                                          self.test_queue, mode='testing')
            self.test_task.start()
            self.master.after(100, self.process_container_queue)

    def train_net(self):
        """
        Method implementing the training process:
        - write the configuration to disk
        - Run the process on a new thread
        """

        if self.param_net_name.get() == 'None':
            print "ERROR: Please, define network name before starting...\n"
            return

        self.trainingBtn['state'] = 'disable'

        if self.train_task is None:
            self.trainingBtn.update()
            self.write_user_configuration()
            print "\n-----------------------"
            print "Running configuration:"
            print "-----------------------"
            print "Train model:", self.param_net_name.get()
            print "Training folder:", self.param_training_folder.get(), "\n"

            print "Method info:"
            print "------------"

            self.train_task = ThreadedTask(self.write_to_console,
                                           self.test_queue,
                                           mode='training')
            self.train_task.start()
            self.master.after(100, self.process_container_queue)

    def check_update(self):
            """
            check update version and propose to download it if differnt
            So far, a rudimentary mode is used to check the last version.
            """

            # I have to discard possible local changes :(
            print "---------------------------------------"
            print "Updating software"
            print "current version:", self.commit_version

            remote_commit = subprocess.check_output(['git', 'stash'])
            remote_commit = subprocess.check_output(['git', 'fetch'])
            remote_commit = subprocess.check_output(['git',
                                                     'rev-parse',
                                                     'origin/master'])

            if remote_commit != self.commit_version:
                proc = subprocess.check_output(['git', 'pull',
                                                'origin', 'master'])
                self.check_link.config(text="Updated")
                self.commit_version = remote_commit
                print "updated version:", self.commit_version
            else:
                print "This software is already in the latest version"
            print "---------------------------------------"

    def about_window(self):
        """
        Window showing information about the software and
        version number, including auto-update. If the application
        is run from a container, then auto-update is disabled
        """

        def callback(event):
            """
            open webbrowser when clicking links
            """
            webbrowser.open_new(event.widget.cget("text"))

        # main window
        t = Toplevel(self.master, width=500, height=500)
        t.wm_title("About")

        # NIC logo + name
        title = Label(t,
                      text="nicMSlesions v" + self.version + "\n"
                      "Multiple Sclerosis White Matter Lesion Segmentation")
        title.grid(row=2, column=1, padx=20, pady=10)
        img = ImageTk.PhotoImage(Image.open('./logonic.png'))
        imglabel = Label(t, image=img)
        imglabel.image = img
        imglabel.grid(row=1, column=1, padx=10, pady=10)
        group_name = Label(t,
                           text="Copyright Sergi Valverde (2019-) \n " +
                           "NeuroImage Computing Group")
        group_name.grid(row=3, column=1)
        group_link = Label(t, text=r"http://atc.udg.edu/nic",
                           fg="blue",
                           cursor="hand2")
        group_link.grid(row=4, column=1)
        group_link.bind("<Button-1>", callback)

        license_content = "Licensed under the BSD 2-Clause license. \n" + \
                          "A copy of the license is present in the root directory."

        license_label = Label(t, text=license_content)
        license_label.grid(row=5, column=1, padx=20, pady=20)

        if self.container is False:
             # check version and updates
             version_number = Label(t, text="commit: " + self.commit_version)
             version_number.grid(row=6, column=1, padx=20, pady=(1, 1))

             self.check_link = Button(t,
                                 text="Check for updates",
                                 command=self.check_update)
             self.check_link.grid(row=7, column=1)

    def process_container_queue(self):
        """
        Process the threading queue. When the threaded processes are
        finished, buttons are reset and a message is shown in the app.
        """
        self.process_indicator.set('Running... please wait')
        try:
            msg = self.test_queue.get(0)
            self.process_indicator.set('Done. See log for more details.')
            self.inferenceBtn['state'] = 'normal'
            self.trainingBtn['state'] = 'normal'
        except Queue.Empty:
            self.master.after(100, self.process_container_queue)

    def close_event(self):
        """
        Stop the thread processes using OS related calls.
        """
        if self.train_task is not None:
            self.train_task.stop_process()
        if self.test_task is not None:
            self.test_task.stop_process()
        os.system('cls' if platform.system == "Windows" else 'clear')
        root.destroy()


class ThreadedTask(threading.Thread):
    """
    Class implementing a threding process (training or inference)
    - train network
    - infer segmentation
    - stop process
    """
    def __init__(self, print_func, queue, mode):
        threading.Thread.__init__(self)
        self.queue = queue
        self.mode = mode
        self.print_func = print_func
        self.process = None

    def run(self):
        """
        Call either the training and testing scripts in cnn_scripts.py.
        """
        options = get_config()
        if self.mode == 'training':
            train_network(options)
        else:
            infer_segmentation(options)
        self.queue.put(" ")

    def stop_process(self):
        """
        stops a parent process and all child processes
        OS dependant
        """
        try:
            if platform.system() == "Windows" :
                subprocess.Popen("taskkill /F /T /PID %i" % os.getpid() , shell=True)
            else:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
        except:
            os.kill(os.getpid(), signal.SIGTERM)


if __name__ == '__main__':
    """
    main script. Check if the method is run inside a docker and then
    call the main application

    python app.py

    options:
    --docker: set if run inside a docker (default is False)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--docker',
                        dest='docker',
                        action='store_true')
    parser.set_defaults(docker=False)
    args = parser.parse_args()
    root = Tk()
    root.resizable(width=False, height=False)
    my_guy = wm_seg(root, args.docker)
    root.mainloop()
