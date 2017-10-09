#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions used for training a CNN."""

import warnings
import numpy as np
import h5py
import os
import keras as ks

#------------- Functions used for supplying images to the GPU -------------#

def generate_batches_from_hdf5_file(filepath, batchsize, n_bins, class_type, f_size=None, zero_center_image=None):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    :param string filepath: Full filepath of the input h5 file, e.g. '/path/to/file/file.h5'.
    :param int batchsize: Size of the batches that should be generated. Ideally same as the chunksize in the h5 file.
    :param tuple n_bins: Number of bins for each dimension (x,y,z,t) in the h5 file.
    :param (int, str) class_type: Tuple with the umber of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :param int f_size: Specifies the filesize (#images) of the .h5 file if not the whole .h5 file
                       but a fraction of it (e.g. 10%) should be used for yielding the xs/ys arrays.
                       This is important if you run fit_generator(epochs>1) with a filesize (and hence # of steps) that is smaller than the .h5 file.
    :param ndarray zero_center_image: mean_image of the x dataset used for zero-centering.
    :return: (ndarray, ndarray) (xs, ys): Yields a tuple which contains a full batch of images and labels.
    """
    dimensions = get_dimensions_encoding(n_bins, batchsize)

    while 1:
        f = h5py.File(filepath, "r")
        if f_size is None:
            f_size = len(f['y'])
            warnings.warn('f_size=None could produce unexpected results if the f_size used in fit_generator(steps=int(f_size / batchsize)) with epochs > 1 '
                          'is not equal to the f_size of the true .h5 file. Should be ok if you use the tb_callback.')

        n_entries = 0
        while n_entries <= (f_size - batchsize):
            # start the next batch at index 0
            # create numpy arrays of input data (features)
            xs = f['x'][n_entries : n_entries + batchsize]
            xs = np.reshape(xs, dimensions).astype(np.float32)

            if zero_center_image is not None: xs = np.subtract(xs, zero_center_image)
            # and mc info (labels)
            y_values = f['y'][n_entries:n_entries+batchsize]
            y_values = np.reshape(y_values, (batchsize, y_values.shape[1]))
            ys = np.zeros((batchsize, class_type[0]), dtype=np.float32)
            # encode the labels such that they are all within the same range (and filter the ones we don't want for now)
            # TODO could be vectorized if performance is a bottleneck. Or just use dataflow from tensorpack!
            for c, y_val in enumerate(y_values):
                ys[c] = encode_targets(y_val, class_type)

            # we have read one more batch from this file
            n_entries += batchsize

            yield (xs, ys)
        f.close() #this line of code is actually not reached if steps=f_size/batchsize


def get_dimensions_encoding(n_bins, batchsize):
    """
    Returns a dimensions tuple for 2,3 and 4 dimensional data.
    :param int batchsize: Batchsize that is used in generate_batches_from_hdf5_file().
    :param tuple n_bins: Declares the number of bins for each dimension (x,y,z).
                        If a dimension is equal to 1, it means that the dimension should be left out.
    :return: tuple dimensions: 2D, 3D or 4D dimensions tuple (integers).
    """
    n_bins_x, n_bins_y, n_bins_z, n_bins_t = n_bins[0], n_bins[1], n_bins[2], n_bins[3]
    if n_bins_x == 1:
        if n_bins_y == 1:
            print 'Using 2D projected data without dimensions x and y'
            dimensions = (batchsize, n_bins_z, n_bins_t, 1)
        elif n_bins_z == 1:
            print 'Using 2D projected data without dimensions x and z'
            dimensions = (batchsize, n_bins_y, n_bins_t, 1)
        elif n_bins_t == 1:
            print 'Using 2D projected data without dimensions x and t'
            dimensions = (batchsize, n_bins_y, n_bins_z, 1)
        else:
            print 'Using 3D projected data without dimension x'
            dimensions = (batchsize, n_bins_y, n_bins_z, n_bins_t, 1)

    elif n_bins_y == 1:
        if n_bins_z == 1:
            print 'Using 2D projected data without dimensions y and z'
            dimensions = (batchsize, n_bins_x, n_bins_t, 1)
        elif n_bins_t == 1:
            print 'Using 2D projected data without dimensions y and t'
            dimensions = (batchsize, n_bins_x, n_bins_z, 1)
        else:
            print 'Using 3D projected data without dimension y'
            dimensions = (batchsize, n_bins_x, n_bins_z, n_bins_t, 1)

    elif n_bins_z == 1:
        if n_bins_t == 1:
            print 'Using 2D projected data without dimensions z and t'
            dimensions = (batchsize, n_bins_x, n_bins_y, 1)
        else:
            print 'Using 3D projected data without dimension z'
            dimensions = (batchsize, n_bins_x, n_bins_y, n_bins_t, 1)

    elif n_bins_t == 1:
        print 'Using 3D projected data without dimension t'
        dimensions = (batchsize, n_bins_x, n_bins_y, n_bins_z, 1)

    else:
        print 'Using full 4D data'
        dimensions = (batchsize, n_bins_x, n_bins_y, n_bins_z, n_bins_t)

    return dimensions


def encode_targets(y_val, class_type):
    """
    Encodes the labels (classes) of the images.
    :param ndarray(ndim=1) y_val: Array that contains ALL event class information for one event.
           ---------------------------------------------------------------------------------------------------------------------------
           Current content: [event_id -> 0, particle_type -> 1, energy -> 2, isCC -> 3, bjorkeny -> 4, dir_x/y/z -> 5/6/7, time -> 8]
           ---------------------------------------------------------------------------------------------------------------------------
    :param (int, str) class_type: Tuple with the umber of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :return: ndarray(ndim=1) train_y: Array that contains the encoded class label information of the input event.
    """
    def get_class_up_down_categorical(dir_z, n_neurons):
        """
        Converts the zenith information (dir_z) to a binary up/down value.
        :param float32 dir_z: z-direction of the event_track (which contains dir_z).
        :param int n_neurons: defines the number of neurons in the last cnn layer that should be used with the categorical array.
        :return ndarray(ndim=1) y_cat_up_down: categorical y ('label') array which can be fed to a NN.
                                               E.g. [0],[1] for n=1 or [0,1], [1,0] for n=2
        """
        # analyze the track info to determine the class number
        up_down_class_value = int(np.sign(dir_z)) # returns -1 if dir_z < 0, 0 if dir_z==0, 1 if dir_z > 0

        if up_down_class_value == 0:
            print 'Warning: Found an event with dir_z==0. Setting the up-down class randomly.'
            #TODO maybe [0.5, 0.5], but does it make sense with cat_crossentropy?
            up_down_class_value = np.random.randint(2)

        if up_down_class_value == -1: up_down_class_value = 0 # Bring -1,1 values to 0,1

        y_cat_up_down = np.zeros(n_neurons, dtype='float32')

        if n_neurons == 1:
            y_cat_up_down[0] = up_down_class_value # 1 or 0 for up/down
        else:
            y_cat_up_down[up_down_class_value] = 1 # [0,1] or [1,0] for up/down

        return y_cat_up_down


    def convert_particle_class_to_categorical(particle_type, is_cc, num_classes=4):
        """
        Converts the possible particle types (elec/muon/tau , NC/CC) to a categorical type that can be used as tensorflow input y
        :param int particle_type: Specifies the particle type, i.e. elec/muon/tau (12, 14, 16). Negative values for antiparticles.
        :param int is_cc: Specifies the interaction channel. 0 = NC, 1 = CC.
        :param int num_classes: Specifies the total number of classes that will be discriminated later on by the CNN. I.e. 2 = elec_NC, muon_CC.
        :return: ndarray(ndim=1) categorical: returns the categorical event type. I.e. (particle_type=14, is_cc=1) -> [0,0,1,0] for num_classes=4.
        """
        if num_classes == 4:
            particle_type_dict = {(12, 0): 0, (12, 1): 1, (14, 1): 2, (16, 1): 3}  # 0: elec_NC, 1: elec_CC, 2: muon_CC, 3: tau_CC
        else:
            raise ValueError('A number of classes !=4 is currently not supported!')

        category = int(particle_type_dict[(abs(particle_type), is_cc)])
        categorical = np.zeros(num_classes, dtype='int8') # TODO try bool
        categorical[category] = 1

        return categorical


    if class_type[1] == 'muon-CC_to_elec-NC':
        categorical_type = convert_particle_class_to_categorical(y_val[1], y_val[3], num_classes=4)
        train_y = np.zeros(class_type[0], dtype='float32') # 1 ([0], [1]) or 2 ([0,1], [1,0]) neurons

        if class_type[0] == 1: # 1 neuron
            if categorical_type[2] != 0:
                train_y[0] = categorical_type[2] # =0 if elec-NC, =1 if muon-CC

        else: # 2 neurons
            assert class_type[0] == 2
            train_y[0] = categorical_type[0]
            train_y[1] = categorical_type[2]

    elif class_type[1] == 'muon-CC_to_elec-CC':
        categorical_type = convert_particle_class_to_categorical(y_val[1], y_val[3], num_classes=4)
        train_y = np.zeros(class_type[0], dtype='float32')

        if class_type[0] == 1: # 1 neuron
            if categorical_type[2] != 0:
                train_y[0] = categorical_type[2] # =0 if elec-CC, =1 if muon-CC

        else: # 2 neurons
            assert class_type[0] == 2
            train_y[0] = categorical_type[1]
            train_y[1] = categorical_type[2]

    elif class_type[1] == 'up_down':
        #supports both 1 or 2 neurons at the cnn softmax end
        train_y = get_class_up_down_categorical(y_val[7], class_type[0])

    else:
        print "Class type " + str(class_type) + " not supported!"
        return y_val

    return train_y

#------------- Functions used for supplying images to the GPU -------------#


#------------- Functions for preprocessing -------------#

def load_zero_center_data(train_files, batchsize, n_bins):
    """
    Gets the xs_mean array that can be used for zero-centering.
    The array is either loaded from a previously saved file or it is calculated on the fly.
    Currently only works for a single input training file!
    :param list((train_filepath, train_filesize)) train_files: list of tuples that contains the trainfiles and their number of rows.
    :param int batchsize: Batchsize that is being used in the data.
    :param tuple n_bins: Number of bins for each dimension (x,y,z,t) in the tran_file.
    :return: ndarray xs_mean: mean_image of the x dataset. Can be used for zero-centering later on.
    """
    if len(train_files) > 1:
        raise Exception('More than 1 train file for zero-centering is currently not supported!')

    filepath = train_files[0][0]

    if os.path.isfile(filepath + '_zero_center_mean.npy') is True:
        print 'Loading an existing xs_mean_array in order to zero_center the data!'
        xs_mean = np.load(filepath + '_zero_center_mean.npy')

    else:
        print 'Calculating the xs_mean_array in order to zero_center the data! Warning: Memory must be as large as the inputfile!'
        dimensions = get_dimensions_encoding(n_bins, batchsize)
        xs_mean = get_mean_image(filepath, dimensions)

    return xs_mean


def get_mean_image(filepath, dimensions):
    """
    Returns the mean_image of a xs dataset by loading or calculating it.
    :param str filepath: Filepath of the data upon which the mean_image should be calculated.
    :param tuple dimensions: dimensions tuple for 2D, 3D or 4D data.
    :param filepath: filepath of the input data, used as a str for saving the xs_mean_image.
    :return: ndarray xs_mean: mean_image of the x dataset. Can be used for zero-centering later on.
    """
    f = h5py.File(filepath, "r")
    # Example: f['x'] has shape (batchsize * x * y * z * 1) #TODO possibly doesn't work for 4D (or 3.5D) data yet!
    # maybe astype np.float64 for increased precision
    xs_mean = np.mean(f['x'], axis=0) # has shape (x * y * z * channels)
    #assert xs_mean.shape == dimensions[1:] # sanity check
    xs_mean = np.reshape(xs_mean, dimensions[1:]) # give the shape the channels dimension again
    #xs_std = np.std(f['x'], axis=0, dtype=np.float64)
    np.save(filepath + '_zero_center_mean.npy', xs_mean)

    return xs_mean

#------------- Functions for preprocessing -------------#


#------------- Various other functions -------------#

def get_modelname(n_bins, class_type):
    """
    Derives the name of a model based on its number of bins and the class_type tuple.
    The final modelname is defined as 'model_Nd_proj_class_type[1]'.
    E.g. 'model_3d_xyz_muon-CC_to_elec-CC'.
    :param tuple n_bins: Number of bins for each dimension (x,y,z,t) of the training images.
    :param (int, str) class_type: Tuple that declares the number of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :return: str modelname: Derived modelname.
    """
    modelname = 'model_'

    dim = 4- n_bins.count(1)

    projection = ''
    if n_bins[0] > 1: projection += 'x'
    if n_bins[1] > 1: projection += 'y'
    if n_bins[2] > 1: projection += 'z'
    if n_bins[3] > 1: projection += 't'

    modelname += str(dim) + 'd_' + projection + '_' + class_type[1]

    return modelname

#------------- Various other functions -------------#


#------------- Classes -------------#

class TensorBoardWrapper(ks.callbacks.TensorBoard):
    """Up to now (05.10.17), Keras doesn't accept TensorBoard callbacks with validation data that is fed by a generator.
     Supplying the validation data is needed for the histogram_freq > 1 argument in the TB callback.
     Without a workaround, only scalar values (e.g. loss, accuracy) and the computational graph of the model can be saved.

     This class acts as a Wrapper for the ks.callbacks.TensorBoard class in such a way,
     that the whole validation data is put into a single array by using the generator.
     Then, the single array is used in the validation steps. This workaround is experimental!"""

    def __init__(self, batch_gen, nb_steps, **kwargs):
        super(TensorBoardWrapper, self).__init__(**kwargs)
        self.batch_gen = batch_gen # The generator.
        self.nb_steps = nb_steps   # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in xrange(self.nb_steps):
            ib, tb = next(self.batch_gen)
            if imgs is None and tags is None:
                imgs = np.zeros(((self.nb_steps * ib.shape[0],) + ib.shape[1:]), dtype=np.float32)
                tags = np.zeros(((self.nb_steps * tb.shape[0],) + tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
        return super(TensorBoardWrapper, self).on_epoch_end(epoch, logs)

#------------- Classes -------------#