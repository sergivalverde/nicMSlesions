
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
import platform
import subprocess
import os
import signal
import Queue
import threading
from Tkinter import Frame, LabelFrame, Label, END, Tk
from Tkinter import Entry, Button, Checkbutton, OptionMenu, Toplevel
from Tkinter import BooleanVar, StringVar, IntVar, DoubleVar
from tkFileDialog import askdirectory
from ttk import Notebook
from PIL import Image, ImageTk
import webbrowser
from cnn_scripts import train_network, infer_segmentation, get_config


class wm_seg:
    def __init__(self, master):

        self.master = master
        # master.minsize(width=100, height=100)
        master.title("nicMSlesions")

        # gui attributes
        self.path = os.getcwd()
        self.default_config = None
        self.user_config = None
        self.current_folder = os.getcwd()
        self.list_train_pretrained_nets = []
        self.list_test_nets = []

        # version_number
        self.commit_version = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'])

        # queue and thread parameters
        self.train_task = None
        self.test_task = None
        self.test_queue = Queue.Queue()
        self.train_queue = Queue.Queue()

        # data parameters
        self.param_training_folder = StringVar()
        self.param_test_folder = StringVar()
        self.param_FLAIR_tag = StringVar()
        self.param_T1_tag = StringVar()
        self.param_mask_tag = StringVar()
        self.param_model_tag = StringVar()
        self.param_register_modalities = BooleanVar()
        self.param_skull_stripping = BooleanVar()
        self.param_save_tmp = BooleanVar()
        self.param_debug = BooleanVar()

        # train parameters
        self.param_net_folder = os.path.join(self.current_folder, 'nets')
        self.param_use_pretrained_model = BooleanVar()
        self.param_pretrained_model = StringVar()
        self.param_inference_model = StringVar()
        self.param_num_layers = IntVar()

        # model parameters
        self.param_net_name = None
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
        self.param_mode = StringVar()

        # load the default configuration from the conf file
        self.load_default_configuration()

        # notebook
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
        print "# (c) Sergi Valverde 2017                        #"
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
        # self.show_frame = LabelFrame(self.train_frame, text="Output:")
        # self.show_frame.grid(row=7, columnspan=cl_s,
        #                      sticky='WE',
        #                     padx=5, pady=5, ipadx=5, ipady=5)

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

        self.flairTagLbl = Label(self.tr_frame, text="FLAIR tag:")
        self.flairTagLbl.grid(row=2, column=0, sticky='E', padx=5, pady=2)
        self.flairTxt = Entry(self.tr_frame,
                              textvariable=self.param_FLAIR_tag)
        self.flairTxt.grid(row=2, column=1, columnspan=1, sticky="W", pady=1)

        self.t1TagLbl = Label(self.tr_frame, text="T1 tag:")
        self.t1TagLbl.grid(row=3, column=0, sticky='E', padx=5, pady=2)
        self.t1Txt = Entry(self.tr_frame, textvariable=self.param_T1_tag)
        self.t1Txt.grid(row=3, column=1, columnspan=1, sticky="W", pady=1)

        self.maskTagLbl = Label(self.tr_frame, text="MASK tag:")
        self.maskTagLbl.grid(row=4, column=0,
                             sticky='E', padx=5, pady=2)
        self.maskTxt = Entry(self.tr_frame, textvariable=self.param_mask_tag)
        self.maskTxt.grid(row=4, column=1, columnspan=1, sticky="W", pady=1)

        # model options
        self.modelTagLbl = Label(self.model_frame, text="Model name:")
        self.modelTagLbl.grid(row=5, column=0,
                              sticky='E', padx=5, pady=2)
        self.modelTxt = Entry(self.model_frame,
                              textvariable=self.param_model_tag)
        self.modelTxt.grid(row=5, column=1, columnspan=1, sticky="W", pady=1)

        self.checkPretrain = Checkbutton(self. model_frame,
                                         text="use pretrained",
                                         var=self.param_use_pretrained_model)
        self.checkPretrain.grid(row=5, column=3, padx=5, pady=5)

        self.update_pretrained_nets()

        self.pretrainTxt = OptionMenu(self.model_frame,
                                      self.param_pretrained_model,
                                      *self.list_train_pretrained_nets)
        self.pretrainTxt.grid(row=5, column=5, sticky='E', padx=5, pady=5)

        # text options
        # self.command_out = Text(self.show_frame)
        # self.command_out.grid(row=7, columnspan=7, sticky='W',
        #                      padx=1, pady=1,
        #                      ipadx=1, ipady=1)

        # START button links to docker task
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
        # self.test_show_frame = LabelFrame(self.test_frame, text="Output:")
        # self.test_show_frame.grid(row=7, columnspan=cl_s,
        #                          sticky='WE',
        #                          padx=5, pady=5, ipadx=5, ipady=5)

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
        self.test_flairTagLbl.grid(row=2, column=0, sticky='E', padx=5, pady=2)
        self.test_flairTxt = Entry(self.tt_frame,
                                   textvariable=self.param_FLAIR_tag)
        self.test_flairTxt.grid(row=2,
                                column=1,
                                columnspan=1,
                                sticky="W",
                                pady=1)

        self.test_t1TagLbl = Label(self.tt_frame, text="T1 tag:")
        self.test_t1TagLbl.grid(row=3, column=0, sticky='E', padx=5, pady=2)
        self.test_t1Txt = Entry(self.tt_frame, textvariable=self.param_T1_tag)
        self.test_t1Txt.grid(row=3, column=1, columnspan=1, sticky="W", pady=1)

        self.test_maskTagLbl = Label(self.tt_frame, text="MASK tag:")
        self.test_maskTagLbl.grid(row=4, column=0,
                                  sticky='E', padx=5, pady=2)
        self.test_maskTxt = Entry(self.tt_frame,
                                  textvariable=self.param_mask_tag)
        self.test_maskTxt.grid(row=4,
                               column=1,
                               columnspan=1,
                               sticky="W",
                               pady=1)

        self.test_pretrainTxt = OptionMenu(self.test_model_frame,
                                           self.param_inference_model,
                                           *self.list_test_nets)

        self.test_pretrainTxt.grid(row=5, column=0, sticky='E', padx=5, pady=5)

        # text options
        #self.test_command_out = Text(self.test_show_frame)
        # self.test_command_out.grid(row=7, columnspan=7, sticky='W',
        #                           padx=1, pady=1,
        #                           ipadx=1, ipady=1)
        # START button links to docker task
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

        # docker state and so on...
        self.process_indicator = StringVar()
        self.process_indicator.set(' ')
        self.label_indicator = Label(master,
                                     textvariable=self.process_indicator)
        self.label_indicator.pack(side="left")

        # master protocol to close things
        self.master.protocol("WM_DELETE_WINDOW", self.close_event)

    def parameter_window(self):
        """
        setting other parameters

        """
        t = Toplevel(self.master)
        t.wm_title("Other parameters")

        # data parameters
        t_data = LabelFrame(t, text="data options:")
        # t_data.pack(fill="both")
        t_data.grid(row=0, sticky="WE")
        checkPretrain = Checkbutton(t_data,
                                    text="Register modalities",
                                    var=self.param_register_modalities)
        checkPretrain.grid(row=0, sticky='W')
        checkSkull = Checkbutton(t_data,
                                 text="Skull-strip modalities",
                                 var=self.param_skull_stripping)
        checkSkull.grid(row=1, sticky="W")
        check_tmp = Checkbutton(t_data,
                                text="Save tmp files",
                                var=self.param_save_tmp)
        check_tmp.grid(row=2, sticky="W")
        checkdebug = Checkbutton(t_data,
                                 text="Show debug messages",
                                 var=self.param_debug)
        checkdebug.grid(row=3, sticky="W")

        # model parameters
        t_model = LabelFrame(t, text="Model:")
        t_model.grid(row=5, sticky="EW")
        # layers_label = Label(t_model, text="Num of retrained layers:")
        # layers_label.grid(row=5, sticky='W')
        # layers_entry = Entry(t_model, textvariable=self.param_num_layers)
        # layers_entry.grid(row=5, column=1, sticky="W")

        # minth_label = Label(t_model, text="Min false positive probability:")
        # minth_label.grid(row=6, sticky="W")
        # minth_entry = Entry(t_model, textvariable=self.param_min_th)
        # minth_entry.grid(row=6, column=1, sticky="W")

        maxepochs_label = Label(t_model, text="Max epochs:           ")
        maxepochs_label.grid(row=6, sticky="W")
        maxepochs_entry = Entry(t_model, textvariable=self.param_max_epochs)
        maxepochs_entry.grid(row=6, column=1, sticky="W")

        # patchsize_label = Label(t_model, text="Patch size:")
        # patchsize_label.grid(row=8, sticky="W")
        # patchsize_entry = Entry(t_model, textvariable=self.param_patch_size)
        # patchsize_entry.grid(row=8, column=1, sticky="W")

        trainsplit_label = Label(t_model, text="Validation %:")
        trainsplit_label.grid(row=7, sticky="W")
        trainsplit_entry = Entry(t_model, textvariable=self.param_train_split)
        trainsplit_entry.grid(row=7, column=1, sticky="W")

        batchsize_label = Label(t_model, text="Test batch size:")
        batchsize_label.grid(row=8, sticky="W")
        batchsize_entry = Entry(t_model, textvariable=self.param_batch_size)
        batchsize_entry.grid(row=8, column=1, sticky="W")

        mode_label = Label(t_model, text="Mode:")
        mode_label.grid(row=9, sticky="W")
        mode_entry = Entry(t_model, textvariable=self.param_mode)
        mode_entry.grid(row=9, column=1, sticky="W")

        mode_label = Label(t_model, text="Verbosity:")
        mode_label.grid(row=11, sticky="W")
        mode_entry = Entry(t_model, textvariable=self.param_net_verbose)
        mode_entry.grid(row=11, column=1, sticky="W")

        # model parameters
        t_post = LabelFrame(t, text="Post-processing:")
        t_post.grid(row=12, sticky="EW")
        t_bin_label = Label(t_post, text="Out probability th:")
        t_bin_label.grid(row=13, sticky="W")
        t_bin_entry = Entry(t_post, textvariable=self.param_t_bin)
        t_bin_entry.grid(row=13, column=1, sticky="W")

        l_min_label = Label(t_post, text="Min out region size:")
        l_min_label.grid(row=14, sticky="W")
        l_min_entry = Entry(t_post, textvariable=self.param_l_min)
        l_min_entry.grid(row=14, column=1, sticky="W")

    def load_default_configuration(self):
        """
        load the default configuration from /config/default
        """

        default_config = ConfigParser.SafeConfigParser()
        default_config.read(os.path.join(self.path, 'config', 'default.cfg'))

        # dastaset parameters
        self.param_training_folder.set(default_config.get('database',
                                                          'train_folder'))
        self.param_test_folder.set(default_config.get('database',
                                                      'inference_folder'))
        self.param_FLAIR_tag.set(default_config.get('database', 'flair_tags'))
        self.param_T1_tag.set(default_config.get('database', 't1_tags'))
        self.param_mask_tag.set(default_config.get('database', 'roi_tags'))
        self.param_register_modalities.set(default_config.get('database', 'register_modalities'))
        self.param_skull_stripping.set(default_config.get('database', 'skull_stripping'))
        self.param_save_tmp.set(default_config.get('database', 'save_tmp'))
        self.param_debug.set(default_config.get('database', 'debug'))

        # train parameters
        self.param_use_pretrained_model.set(default_config.get('train', 'full_train'))
        self.param_pretrained_model.set(default_config.get('train', 'pretrained_model'))
        self.param_inference_model.set("      ")
        # self.param_num_layers.set(default_config.get('train', 'num_layers'))

        # model parameters
        self.param_net_folder = os.path.join(self.current_folder, 'nets')
        self.param_model_tag.set(default_config.get('model', 'name'))
        # self.param_pretrained.set(default_config.get('model', 'pretrained'))
        # self.param_min_th.set(default_config.getfloat('model', 'min_th'))
        # self.param_patch_size.set(default_config.getint('model', 'patch_size'))
        # self.param_weight_paths.set(default_config.get('model', 'weight_paths'))
        # self.param_load_weights.set(default_config.get('model', 'load_weights'))
        self.param_train_split.set(default_config.getfloat('model', 'train_split'))
        self.param_max_epochs.set(default_config.getint('model', 'max_epochs'))
        self.param_patience.set(default_config.getint('model', 'patience'))
        self.param_batch_size.set(default_config.getint('model', 'batch_size'))
        self.param_net_verbose.set(default_config.get('model', 'net_verbose'))
        self.param_mode.set(default_config.get('model', 'mode'))

        # post-processing
        self.param_l_min.set(default_config.getint('postprocessing',
                                                   'l_min'))
        self.param_t_bin.set(default_config.getfloat('postprocessing',
                                                     't_bin'))
                
    def write_user_configuration(self):
        """
        write the configuration into a file
        """
        user_config = ConfigParser.RawConfigParser()
        # dastaset parameters
        user_config.add_section('database')
        user_config.set('database', 'train_folder', self.param_training_folder.get())
        user_config.set('database', 'inference_folder', self.param_test_folder.get())
        user_config.set('database', 'flair_tags', self.param_FLAIR_tag.get())
        user_config.set('database', 't1_tags', self.param_T1_tag.get())
        user_config.set('database', 'roi_tags', self.param_mask_tag.get())
        user_config.set('database', 'register_modalities', self.param_register_modalities.get())
        user_config.set('database', 'skull_stripping', self.param_skull_stripping.get())
        user_config.set('database', 'save_tmp', self.param_save_tmp.get())
        user_config.set('database', 'debug', self.param_debug.get())

        # train parameters
        user_config.add_section('train')
        user_config.set('train', 'full_train', not(self.param_use_pretrained_model.get()))
        user_config.set('train', 'pretrained_model', self.param_pretrained_model.get())
        # user_config.set('train', 'num_layers', self.param_num_layers.get())

        # model parameters
        user_config.add_section('model')
        user_config.set('model', 'name', self.param_model_tag.get())
        user_config.set('model', 'pretrained', self.param_pretrained)
        # user_config.set('model', 'min_th', self.param_min_th.get())
        # user_config.set('model', 'patch_size', self.param_patch_size.get())
        # user_config.set('model', 'weight_paths', self.param_weight_paths.get())
        # user_config.set('model', 'load_weights', self.param_load_weights.get())
        user_config.set('model', 'train_split', self.param_train_split.get())
        user_config.set('model', 'max_epochs', self.param_max_epochs.get())
        user_config.set('model', 'patience', self.param_patience.get())
        user_config.set('model', 'batch_size', self.param_batch_size.get())
        user_config.set('model', 'net_verbose', self.param_net_verbose.get())
        user_config.set('model', 'mode', self.param_mode.get())

        # postprocessing parameters
        user_config.add_section('postprocessing')
        user_config.set('postprocessing', 't_bin', self.param_t_bin.get())
        user_config.set('postprocessing', 'l_min', self.param_l_min.get())

        # Writing our configuration file to 'example.cfg'
        with open(os.path.join(self.path,
                               'config',
                               'configuration.cfg'), 'wb') as configfile:
            user_config.write(configfile)

    def load_training_path(self):
        """
        load the training path from disk
        """
        fname = askdirectory()
        #self.write_to_console(fname + '\n')
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
        load the test path from disk
        """
        fname = askdirectory()
        #self.write_to_console(fname + '\n')
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
        get the different nets present to be used
        """
        folders = os.listdir(self.param_net_folder)
        self.list_train_pretrained_nets = folders + ['baseline']
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
        to doc
        test funcionality of the docker container insed
        - write the configuration to disk
        - open the docker container
        - print
        """

        net_state = self.param_inference_model.get()
        if net_state == '      ':
            self.write_to_test_console("ERROR: Please, select a network model ...\n")
            return
        if self.test_task is None:
            self.inferenceBtn.config(state='disabled')
            self.param_model_tag.set(self.param_inference_model.get())
            self.param_use_pretrained_model.set(False)
            self.write_user_configuration()
            # self.write_to_test_console('-----------------------------------' +
            # self.write_to_test_console('Configuration:' + '\n')
            # self.write_to_test_console('Testing folder: ' +
            #                           self.param_test_folder.get() + '\n')
            # self.write_to_test_console('Net model: ' +
            #                           self.param_model_tag.get() + '\n')
            # self.write_to_test_console('-----------------------------------' +
            #                           '\n\n')
            #                           print "\n\n-------------------------------------------"
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
        - write the configuration to disk
        - open the docker container
        - print
        """
        self.trainingBtn['state'] = 'disable'
        if self.train_task is None:
            self.trainingBtn.update()
            self.write_user_configuration()
            # self.write_to_console('-----------------------------------' + '\n')
            # self.write_to_console('Configuration:' + '\n')
            # self.write_to_console('Train model: ' +
            #                       self.param_model_tag.get() + '\n')
            # self.write_to_console('Training folder: ' +
            #                       self.param_training_folder.get() + '\n')
            # self.write_to_console('-----------------------------------' +
            #                       '\n\n')
            print "\n-----------------------"
            print "Running configuration:"
            print "-----------------------"
            print "Train model:", self.param_model_tag.get()
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
            check update version and propose to download it

            So far, a rudimentary mode is used to check the last version.
            
            """
            # I have to force possible changes in code :(

            print "---------------------------------------"
            print "Updating software"
            print "current version:", self.commit_version
            
            remote_commit = subprocess.check_output(['git', 'stash'])
            remote_commit = subprocess.check_output(['git', 'fetch'])
            remote_commit = subprocess.check_output(['git',
                                                     'rev-parse',
                                                     'origin/master'])

            if remote_commit != self.commit_version:
                proc = subprocess.check_output(['git', 'pull', 'origin', 'master'])
                self.check_link.config(text="Updated")
                self.commit_version = remote_commit
                print "updated version:", self.commit_version
            else:
                print "This software is already in the latest version"
            print "---------------------------------------"
    def about_window(self): 
        """
        About window
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
                      text="nicMSlesions\n " +
                      "Multiple Sclerosis White Matter Lesion Segmentation")
        title.grid(row=2, column=1, padx=20, pady=10)
        img = ImageTk.PhotoImage(Image.open('./logonic.png'))
        imglabel = Label(t, image=img)
        imglabel.image = img
        imglabel.grid(row=1, column=1, padx=10, pady=10)
        group_name = Label(t,
                           text="Copyright Sergi Valverde (2017-) \n " +
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

        # check version and updates
        version_number = Label(t, text="commit: " + self.commit_version)
        version_number.grid(row=6, column=1, padx=20, pady=(1, 1))

        self.check_link = Button(t,
                            text="Check for updates",
                            command=self.check_update)
        self.check_link.grid(row=7, column=1)

    def process_container_queue(self):
        """
        to doc
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
        to doc
        """
        if self.train_task is not None:
            self.train_task.stop_process()
        if self.test_task is not None:
            self.test_task.stop_process()
        os.system('cls' if platform.system == "Windows" else 'clear')
        root.destroy()

'''
class ThreadedTask(threading.Thread):
    """
    to doc
    """
    def __init__(self, print_func, queue, mode):
        threading.Thread.__init__(self)
        self.queue = queue
        self.mode = mode
        self.print_func = print_func
        self.process = None

    def run(self):
        """
        to doc
        """
        if self.mode == 'training':
            filename = 'train.log'
            with io.open(filename, 'wb') as writer, io.open(filename, 'rb', 1) as reader:
                self.process = subprocess.Popen(['python', 'train_network.py'],
                                                stdout=writer, shell=False)
                while self.process.poll() is None:
                    self.print_func(reader.read().decode('utf8'))
                    time.sleep(1)

                # Read the remaining
                sys.stdout.write(reader.read())
                sys.stdout.flush()
        else:
            filename = 'test.log'
            with io.open(filename, 'wb') as writer, io.open(filename, 'rb', 1) as reader:
                self.process = subprocess.Popen(['python',
                                                 'infer_segmentation_batch.py'],
                                                stdout=writer,
                                                shell=False)
                while self.process.poll() is None:
                    self.print_func(reader.read().decode('utf8'))
                    time.sleep(1)
                # Read the remaining
                sys.stdout.write(reader.read())

        sys.stdout.flush()
        self.queue.put(" ")
'''
        

class ThreadedTask(threading.Thread):
    """
    to doc
    """
    def __init__(self, print_func, queue, mode):
        threading.Thread.__init__(self)
        self.queue = queue
        self.mode = mode
        self.print_func = print_func
        self.process = None

    def run(self):
        """
        to doc
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
        """
        try:
            if platform.system() == "Windows" :
                subprocess.Popen("taskkill /F /T /PID %i" % os.getpid() , shell=True)  
            else:     
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
        except:
            os.kill(os.getpid(), signal.SIGTERM)


if __name__ == '__main__':
    root = Tk()
    root.resizable(width=False, height=False)
    my_guy = wm_seg(root)
    root.mainloop()
