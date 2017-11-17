import os, re, gc
import numpy as np
from scipy import ndimage as nd
from nibabel import load as load_nii
import nibabel as nib 
from math import floor
from operator import itemgetter
import cPickle as pickle
import copy
from build_model_nolearn import define_training_layers
from operator import add


def train_cascaded_model(model, train_x_data, train_y_data, options):
    """
    Train the model using a cascade of two CNN

    inputs:
      
    - CNN model: a list containing the two cascaded CNN models 

    - train_x_data: a nested dictionary containing training image paths: 
           train_x_data['scan_name']['modality'] = path_to_image_modality

    - train_y_data: a dictionary containing labels 
        train_y_data['scan_name'] = path_to_label

    - options: dictionary containing general hyper-parameters:
    

    Outputs:
        - trained model: list containing the two cascaded CNN models after training 
    """

    # ----------
    # CNN1
    # ----------

    max_epochs = options['max_epochs']
    
    print "> CNN: loading training data for first model"
    X, Y = load_training_data(train_x_data, train_y_data, options)
    print '> CNN: train_x ', X.shape

    # If a full train is not selected, all CONV layers are freezed and negatives samples
    # are resampled to increase the number of negative samples. Resampling is set to 10 by
    # default
    if options['full_train'] is False:
        model[0] = define_training_layers(net = model[0],
                                          num_layers = options['num_layers'],
                                          number_of_samples = X.shape[0])
        
        num_iterations = options['max_epochs'] / 10
        model[0].max_epochs =  10
        for it in range(num_iterations):
            model[0].fit(X, Y)

            # if early stopping happens exit training  
            last_valid_epoch = model[0].on_epoch_finished[2].best_valid_epoch 
            if (last_valid_epoch + options['patience']) < ((it +1)*10):
                break
                
            X, Y = load_training_data(train_x_data, train_y_data, options)
            
    else:
        model[0].fit(X, Y)

    # ----------
    # CNN2
    # ----------

    print '> CNN: loading training data for the second model'
    X, Y = load_training_data(train_x_data, train_y_data, options,  model = model[0])
    print '> CNN: train_x ', X.shape

    # define training layers
    if options['full_train'] is False:
        model[1] = define_training_layers(net = model[1],
                                          num_layers = options['num_layers'],
                                          number_of_samples = X.shape[0])
        
        num_iterations = options['max_epochs'] / 10
        model[1].max_epochs = 10
        for it in range(num_iterations):
            model[1].fit(X, Y)

            # if early stopping happens exit training  
            last_valid_epoch = model[1].on_epoch_finished[2].best_valid_epoch
            if (last_valid_epoch + options['patience']) < ((it +1)*10):
                break
            
            X, Y = load_training_data(train_x_data, train_y_data, options, model = model[0])
    else:
        model[1].fit(X, Y)
    
    model[0].max_epochs = options['max_epochs']
    model[1].max_epochs = options['max_epochs']

    return model


def test_cascaded_model(model, test_x_data, options):
    """
    Test the cascaded approach using a learned model 

    inputs:
      
    - CNN model: a list containing the two cascaded CNN models 

    - test_x_data: a nested dictionary containing testing image paths: 
           test_x_data['scan_name']['modality'] = path_to_image_modality


    - options: dictionary containing general hyper-parameters:
    
    outputs:
        - output_segmentation 
    """

    # print '> CNN: testing the model'

    # organize experiments
    exp_folder = os.path.join(options['test_folder'], options['test_scan'], options['experiment'])
    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)

    # first network
    options['test_name'] = options['experiment'] + '_prob_0.nii.gz'

    # only save the first iteration result if debug is True
    save_nifti = True if options['debug'] is True else False
    t1 = test_scan(model[0], test_x_data, options, save_nifti= save_nifti)

    # second network 
    options['test_name'] = options['experiment'] + '_debug_prob_1.nii.gz'
    t2 = test_scan(model[1], test_x_data, options, save_nifti= True, candidate_mask = t1>0.8)

    # postprocess the output segmentation
    # obtain the orientation from the first scan used for testing
    scans = test_x_data.keys()
    flair_scans = [test_x_data[s]['FLAIR'] for s in scans]
    flair_image = load_nii(flair_scans[0])
    options['test_name'] = options['experiment'] + '_hard_seg.nii.gz'
    out_segmentation = post_process_segmentation(t2,
                                                 options,
                                                 save_nifti = True,
                                                 orientation = flair_image.affine)

    return out_segmentation


def load_training_data(train_x_data, train_y_data, options, model = None):
    '''
    Load training and label samples for all given scans and modalities.

    Inputs: 

    train_x_data: a nested dictionary containing training image paths: 
        train_x_data['scan_name']['modality'] = path_to_image_modality

    train_y_data: a dictionary containing labels 
        train_y_data['scan_name'] = path_to_label

    options: dictionary containing general hyper-parameters:
        - options['min_th'] = min threshold to remove voxels for training
        - options['size'] = tuple containing patch size, either 2D (p1, p2, 1) or 3D (p1, p2, p3)
        - options['randomize_train'] = randomizes data 
        - options['fully_conv'] = fully_convolutional labels. If false, 

    model: CNN model used to select training candidates

    Outputs:
        - X: np.array [num_samples, num_channels, p1, p2, p2]
        - Y: np.array [num_samples, 1, p1, p2, p2] if fully conv, [num_samples, 1] otherwise

    '''
    
    # get_scan names and number of modalities used 
    scans = train_x_data.keys()
    modalities = train_x_data[scans[0]].keys()

    # select voxels for training:
    #   if model is no passed, training samples are extract by discarding CSF and darker WM in FLAIR, and use all remaining voxels.
    #   if model is passes, use the trained model to extract all voxels with probability > 0.5 
    if model is None:
        flair_scans = [train_x_data[s]['FLAIR'] for s in scans]
        selected_voxels = select_training_voxels(flair_scans, options['min_th'])
    else:
        selected_voxels = select_voxels_from_previous_model(model, train_x_data, options)
        
    # extract patches and labels for each of the modalities
    data = []

    for m in modalities:
        x_data = [train_x_data[s][m] for s in scans]
        y_data = [train_y_data[s] for s in scans]
        x_patches, y_patches = load_train_patches(x_data, y_data, selected_voxels, options['patch_size'])
        data.append(x_patches)
    # stack patches in channels [samples, channels, p1, p2, p3]
    X = np.stack(data, axis = 1)
    Y = y_patches

    # apply randomization if selected
    if options['randomize_train']:
        
        seed = np.random.randint(np.iinfo(np.int32).max)
        np.random.seed(seed)
        X = np.random.permutation(X.astype(dtype=np.float32))
        np.random.seed(seed)
        Y = np.random.permutation(Y.astype(dtype=np.int32))

    # fully convolutional / voxel labels
    if options['fully_convolutional']:
        # Y = [ num_samples, 1, p1, p2, p3]
        Y = np.expand_dims(Y, axis = 1)
    else:
        # Y = [num_samples,]
        if Y.shape[3] == 1:
            Y = Y[:, Y.shape[1] / 2, Y.shape[2] / 2, :]
        else:
            Y = Y[:, Y.shape[1] / 2, Y.shape[2] / 2, Y.shape[3] / 2]
        Y = np.squeeze(Y)

    return X, Y


def select_training_voxels(input_masks, threshold=2, datatype=np.float32):
    """
    Select voxels for training based on a intensity threshold

    Inputs:
        - input_masks: list containing all subject image paths for a single modality
        - threshold: minimum threshold to apply (after normalizing images with 0 mean and 1 std)
    
    Output:
        - rois: list where each element contains the subject binary mask for selected voxels [len(x), len(y), len(z)]
    """

    # load images and normalize their intensities
    images = [load_nii(image_name).get_data() for image_name in input_masks]
    images_norm = [(im.astype(dtype=datatype) - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in images]

    # select voxels with intensity higher than threshold
    rois = [image > threshold for image in images_norm]
    return rois


def load_train_patches(x_data, y_data, selected_voxels, patch_size, random_state = 42, datatype=np.float32):
    """
    Load train patches with size equal to patch_size, given a list of selected voxels

    Inputs: 
       - x_data: list containing all subject image paths for a single modality
       - y_data: list containing all subject image paths for the labels
       - selected_voxels: list where each element contains the subject binary mask for selected voxels [len(x), len(y), len(z)]
       - tuple containing patch size, either 2D (p1, p2, 1) or 3D (p1, p2, p3)
    
    Outputs:
       - X: Train X data matrix for the particular channel [num_samples, p1, p2, p3]
       - Y: Train Y labels [num_samples, p1, p2, p3]
    """
    
    # load images and normalize their intensties
    images = [load_nii(name).get_data() for name in x_data]
    images_norm = [(im.astype(dtype=datatype) - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in images]

    # load labels 
    lesion_masks = [load_nii(name).get_data().astype(dtype=np.bool) for name in y_data]
    nolesion_masks = [np.logical_and(np.logical_not(lesion), brain) for lesion, brain in zip(lesion_masks, selected_voxels)]

    # Get all the x,y,z coordinates for each image
    lesion_centers = [get_mask_voxels(mask) for mask in lesion_masks]
    nolesion_centers = [get_mask_voxels(mask) for mask in nolesion_masks]
   
    # load all positive samples (lesion voxels) and the same number of random negatives samples
    np.random.seed(random_state) 

    x_pos_patches = [np.array(get_patches(image, centers, patch_size)) for image, centers in zip(images_norm, lesion_centers)]
    y_pos_patches = [np.array(get_patches(image, centers, patch_size)) for image, centers in zip(lesion_masks, lesion_centers)]
    
    indices = [np.random.permutation(range(0, len(centers1))).tolist()[:len(centers2)] for centers1, centers2 in zip(nolesion_centers, lesion_centers)]
    nolesion_small = [itemgetter(*idx)(centers) for centers, idx in zip(nolesion_centers, indices)]
    x_neg_patches = [np.array(get_patches(image, centers, patch_size)) for image, centers in zip(images_norm, nolesion_small)]
    y_neg_patches = [np.array(get_patches(image, centers, patch_size)) for image, centers in zip(lesion_masks, nolesion_small)]

    # concatenate positive and negative patches for each subject
    X = np.concatenate([np.concatenate([x1, x2]) for x1, x2 in zip(x_pos_patches, x_neg_patches)], axis = 0)
    Y = np.concatenate([np.concatenate([y1, y2]) for y1, y2 in zip(y_pos_patches, y_neg_patches)], axis= 0)
    
    return X, Y


def load_test_patches(test_x_data, patch_size, batch_size, voxel_candidates = None, datatype=np.float32):
    """
    Function generator to load test patches with size equal to patch_size, given a list of selected voxels. Patches are
    returned in batches to reduce the amount of RAM used
    
    Inputs: 
       - x_data: list containing all subject image paths for a single modality
       - selected_voxels: list where each element contains the subject binary mask for selected voxels [len(x), len(y), len(z)]
       - tuple containing patch size, either 2D (p1, p2, 1) or 3D (p1, p2, p3)
       - Voxel candidates: a binary mask containing voxels to select for testing 
    
    Outputs (in batches):
       - X: Train X data matrix for the particular channel [num_samples, p1, p2, p3]
       - voxel_coord: list of tuples corresponding voxel coordinates (x,y,z) of selected patches 
    """

    # get scan names and number of modalities used 
    scans = test_x_data.keys()
    modalities = test_x_data[scans[0]].keys()

    # load all image modalities and normalize intensities 
    images = []


    for m in modalities:
        raw_images = [load_nii(test_x_data[s][m]).get_data() for s in scans]
        images.append([(im.astype(dtype=datatype) - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in raw_images])

    # select voxels for testing. Discard CSF and darker WM in FLAIR.
    # If voxel_candidates is not selected, using intensity > 0.5 in FLAIR, else use
    # the binary mask to extract candidate voxels 
    if voxel_candidates is None:
        flair_scans = [test_x_data[s]['FLAIR'] for s in scans]
        selected_voxels = [get_mask_voxels(mask) for mask in select_training_voxels(flair_scans, 0.5)][0]
    else:
        selected_voxels = get_mask_voxels(voxel_candidates)
    
    # yield data for testing with size equal to batch_size
    for i in range(0, len(selected_voxels), batch_size):
        c_centers = selected_voxels[i:i+batch_size]
        X = []
        for image_modality in images:
            X.append(get_patches(image_modality[0], c_centers, patch_size))
        yield np.stack(X, axis = 1), c_centers


def get_mask_voxels(mask):
    """
    Compute x,y,z coordinates of a binary mask 

    Input: 
       - mask: binary mask
    
    Output: 
       - list of tuples containing the (x,y,z) coordinate for each of the input voxels
    """
    
    indices = np.stack(np.nonzero(mask), axis=1)
    indices = [tuple(idx) for idx in indices]
    return indices

def get_patches(image, centers, patch_size=(15, 15, 15)):
    """
    Get image patches of arbitrary size based on a set of centers
    """
    # If the size has even numbers, the patch will be centered. If not, it will try to create an square almost centered.
    # By doing this we allow pooling when using encoders/unets.
    patches = []
    list_of_tuples = all([isinstance(center, tuple) for center in centers])
    sizes_match = [len(center) == len(patch_size) for center in centers]

    if list_of_tuples and sizes_match:
        patch_half = tuple([idx/2 for idx in patch_size])
        new_centers = [map(add, center, patch_half) for center in centers]
        padding = tuple((idx, size-idx) for idx, size in zip(patch_half, patch_size))
        new_image = np.pad(image, padding, mode='constant', constant_values=0)
        slices = [[slice(c_idx-p_idx, c_idx+(s_idx-p_idx)) for (c_idx, p_idx, s_idx) in zip(center, patch_half, patch_size)] for center in new_centers]
        patches = [new_image[idx] for idx in slices]
        
    return patches


def test_scan(model, test_x_data, options, save_nifti= True, candidate_mask = None):
    """
    Test data based on one model 
    Input: 
    - test_x_data: a nested dictionary containing training image paths: 
            train_x_data['scan_name']['modality'] = path_to_image_modality
    - save_nifti: save image segmentation 
    - candidate_mask: a binary masks containing voxels to classify

    Output:
    - test_scan = Output image containing the probability output segmetnation 
    - If save_nifti --> Saves a nifti file at specified location options['test_folder']/['test_scan']
    """

    # get_scan name and create an empty nifti image to store segmentation
    scans = test_x_data.keys()
    flair_scans = [test_x_data[s]['FLAIR'] for s in scans]
    flair_image = load_nii(flair_scans[0])
    seg_image = np.zeros_like(flair_image.get_data())

    
    # compute lesion segmentation in batches of size options['batch_size'] 
    for batch, centers in load_test_patches(test_x_data, options['patch_size'], options['batch_size'], candidate_mask):
        y_pred = model.predict_proba(np.squeeze(batch))
        [x, y, z] = np.stack(centers, axis=1)
        seg_image[x, y, z] = y_pred[:, 1]

    if save_nifti:
        out_scan = nib.Nifti1Image(seg_image, affine=flair_image.affine)
        out_scan.to_filename(os.path.join(options['test_folder'], options['test_scan'], options['experiment'], options['test_name']))
        #out_scan.to_filename(os.path.join(test_folder, scan, options['experiment'], options['test_name']))

    return seg_image 


def select_voxels_from_previous_model(model, train_x_data, options):
    """
    Select training voxels from image segmentation masks 
    
    """

    # get_scan names and number of modalities used 
    scans = train_x_data.keys()
    modalities = train_x_data[scans[0]].keys()

    # select voxels for training. Discard CSF and darker WM in FLAIR. 
    # flair_scans = [train_x_data[s]['FLAIR'] for s in scans]
    # selected_voxels = select_training_voxels(flair_scans, options['min_th'])
    
    # evaluate training scans using the learned model and extract voxels with probability higher than 0.5
    seg_mask  = [test_scan(model, dict(train_x_data.items()[s:s+1]), options, save_nifti = False) > 0.5 for s in range(len(scans))]

    # check candidate segmentations:
    # if no voxels have been selected, return candidate voxels on FLAIR modality > 2
    flair_scans = [train_x_data[s]['FLAIR'] for s in scans]
    images = [load_nii(name).get_data() for name in flair_scans]
    images_norm = [(im.astype(dtype=np.float32) - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in images]
    seg_mask = [im > 2 if np.sum(seg) == 0 else seg for im, seg in zip(images_norm, seg_mask)]
    
    return seg_mask


def post_process_segmentation(input_scan, options, save_nifti = True, orientation = np.eye(4)):
    """
    Post-process the probabilistic segmentation using parameters t_bin and l_min
    t_bin: threshold to binarize the output segmentations 
    l_min: minimum lesion volume

    Inputs: 
    - input_scan: probabilistic input image (segmentation)
    - options dictionary
    - save_nifti: save the result as nifti 

    Output:
    - output_scan: final binarized segmentation 
    """

    from scipy import ndimage
    
    t_bin = options['t_bin']
    l_min = options['l_min']
    output_scan = np.zeros_like(input_scan)

    # threshold input segmentation
    t_segmentation = input_scan >= t_bin
    
    # filter candidates by size and store those > l_min
    labels, num_labels = ndimage.label(t_segmentation)
    label_list = np.unique(labels)
    num_elements_by_lesion = ndimage.labeled_comprehension(t_segmentation, labels, label_list, np.sum, float, 0)

    for l in range(len(num_elements_by_lesion)):
        if num_elements_by_lesion[l]>l_min:
            # assign voxels to output
            current_voxels = np.stack(np.where(labels == l), axis =1)
            output_scan[current_voxels[:,0], current_voxels[:,1], current_voxels[:,2]] = 1

    #save the output segmentation as Nifti1Image
    if save_nifti:
        nifti_out = nib.Nifti1Image(output_scan, affine = orientation)
        nifti_out.to_filename(os.path.join(options['test_folder'], options['test_scan'], options['experiment'], options['test_name']))

    return output_scan

