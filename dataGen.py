# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# https://github.com/Saqibm128/eeg-tasks/blob/master/keras_models/dataGen.py
import numpy as np
import numpy.random
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from addict import Dict
from numpy.random import choice
import multiprocessing as mp
import time

#Wrapper classes for batch training in Keras

def three_dim_pad(data, mask_value, num_channels=1, max_length=None):
    """Used to pad in a list of variable length data

    Parameters
    ----------
    data : list
        List of data in shape time by features.
    mask_value : float or int
        used as placeholder for masking layer in an LSTM

    Returns
    -------
    np.array
        correctly sized np.array for batch with mask values filled in

    """
    # for n_batch, n_timestep, n_input matrix, pad_sequences fails
    lengths = [datum.shape[0] for datum in data]
    if max_length is None:
        max_length = max(lengths)
    paddedBatch = np.zeros((len(data), max_length, *data[0].shape[1:], num_channels))
    paddedBatch.fill(mask_value)
    for i, datum in enumerate(data):
        if type(datum) == pd.DataFrame:
            datum = datum.values
        if num_channels == 1:
            datum = datum.reshape(*datum.shape, 1)
        if max_length is None:
            paddedBatch[i, 0:lengths[i], :] = datum
        else:
            datum = datum[0:max_length,:]
            paddedBatch[i, 0:datum.shape[0], :] = datum
    return paddedBatch

class DataGenerator(keras.utils.Sequence):
    '''
    Generates data for Keras, based on code from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    Primarily made because I thought data wouldn't fit inside memory
    '''
    def __init__(self, list_IDs, labels, data=None, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.data = data
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_x_y(self, id):
        return np.load('data/' + ID + '.npy'), self.labels[ID]

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,], y[i] = self.get_x_y(ID)

class EdfDataGenerator(DataGenerator):
    'Can accept EdfDataset and any of its intermediates to make data (i.e. sftft)'
    def __init__(self, dataset, mask_value=-10000, labels=None, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, class_type="nominal", use_three_dim_pad=True, shuffle=True, max_length=None, time_first=True, precache=False, xy_tuple_form=True, separate_x_y=False, use_background_process=False):

        super().__init__(list_IDs=list(range(len(dataset))), labels=labels, batch_size=batch_size, dim=dim, n_channels=n_channels,
                     n_classes=n_classes, shuffle=shuffle)
        if not xy_tuple_form:
            self.list_IDs = list(range(len(dataset[0]))) #the dataset is a tuple of x and y. grab x and use that length.
            self.on_epoch_end()
        self.dataset = dataset
        self.mask_value=mask_value
        self.max_length=max_length
        self.time_first = time_first
        self.use_three_dim_pad = use_three_dim_pad
        self.separate_x_y = separate_x_y
        self.xy_tuple_form = xy_tuple_form
        self.class_type=class_type

        if precache: #just populate self.labels too if we are precaching anyways
            self.dataset = dataset[:]
            if self.labels is None:
                self.labels = np.array([datum[1] for datum in self.dataset])
        self.precache = precache
        if type(self.labels) == list:
            self.labels = np.array(self.labels)

    def get_x_y(self, i):
        if self.precache and self.xy_tuple_form:
            data = [self.dataset[j] for j in i]
        elif self.xy_tuple_form:
            data = self.dataset[i]
        if self.xy_tuple_form:
            x = [datum[0] for datum in data]
            if self.labels is not None:
                y = self.labels[i]
            else:
                y = [datum[1] for datum in data]
        elif self.separate_x_y:
            x = []
            y = []
            for j in i:
                x_j = self.dataset[j]
                y_j = self.labels[j]
                x.append(x_j)
                y.append(y_j)
            return np.array(x), np.array(y)
        else:
            x = self.dataset[0][i]
            y = self.dataset[1][i]

        return x, y

    def __getitem__(self, index, accessed_by_background=False):
        'Generate one batch of data'
#         if not accessed_by_background and self.use_background_process:
#             while self.resultQueue.empty():
#                 time.sleep(0.1)
#             return self.resultQueue.get()

        # if self.use_background_process:
            # print(self.resultQueue.qsize())

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def __data_generation(self, list_IDs_temp):
        '''Overriding this to deal with None dimensions (i.e. variable length times)
        and allow for MultiProcessingDataset to work (only works for slices) '''
       
        x, y = self.get_x_y(list_IDs_temp)
        if self.use_three_dim_pad:
            x = three_dim_pad(x, self.mask_value, max_length=self.max_length)
        if not self.time_first: # we want batch by feature by time
            x = x.transpose((0, 2,1, *[i + 3 for i in range(x.ndim - 3)]))
        if not hasattr(self, "class_type") or self.class_type == "nominal":
            y =  keras.utils.to_categorical(y, num_classes=self.n_classes)
        elif self.class_type == "quantile":
            y = y
        elif self.class_type == "custom":
            y = self.transform_y(y)
        return x, np.stack(y)
    
class RandomRearrangeBatchGenerator(keras.utils.Sequence):
    def __init__(self, dataset, axis=1, dataset_has_epoch_end =True):
        self.dataset = dataset
        self.axis = axis
        self.dataset_has_epoch_end = dataset_has_epoch_end
    def on_epoch_end(self):
        if self.dataset_has_epoch_end:
            return self.dataset.on_epoch_end()
    def __getitem__(self, i):
        x, y = self.dataset[i]
        num_channels = x.shape[self.axis]
        rearrangement = np.random.choice(num_channels, num_channels, replace=False)
        x = x[[slice(None) if i != self.axis else rearrangement for i in range(len(x.shape))]]
        return x, y
    def __len__(self):
        return len(self.dataset)
    
class TransposerBatchGenerator(keras.utils.Sequence):
    def __init__(self, dataset, transpose, dataset_has_epoch_end=True):
        self.dataset = dataset
        self.transpose = transpose
        self.dataset_has_epoch_end = dataset_has_epoch_end
    def __getitem__(self, i):
        x, y = self.dataset[i]
        x = x.transpose(self.transpose)
        return x, y
    def on_epoch_end(self):
        if self.dataset_has_epoch_end:
            return self.dataset.on_epoch_end()