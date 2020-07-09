import constants
import pandas as pd
import numpy as np
import itertools
import pyedflib
from os import path
import sys, os
import multiprocessing as mp
import argparse
import pickle as pkl
import re
from scipy.signal import butter, lfilter
import pywt
import filters
from addict import Dict
import time
import functools
from copy import deepcopy
import pickle as pkl
import json
import os
import os.path as path
import pandas as pd
import numpy as np
import pymongo
import itertools
import pyedflib
from sacred.serializer import restore  # to return a stored sacred result back
import multiprocessing as mp
import queue
from functools import lru_cache
import string, random
from scipy.signal import coherence

class MultiProcessingDataset():
    """Class to help improve speed of looking up multiple records at once using multiple processes.
        Was originally going to be designed around batch loading in, but was just used as a way to more quickly
        populate an array-like into memory
            Just make this the parent class, then call the getItemSlice method on slice objects
        Issues:
            Doesn't solve original problem of being optimized for keras batches, only solves
                the fact that I needed some dataset that could quickly use multiple cores to
                get data. Use the models in keras_models.dataGen
            SLURM opaquely kills processes if it consume too much memory, so we gotta
                double check and see that there are placeholders in the toReturn array left
            The toReturn array uses integer placeholders (representing logical indices of the
                dataset ). If the returning datatype returned by indexing is also
                an integer, then this won't work
            Recovery from OOM is single threaded. Maybe we wanna make this
                use mp if this becomes a new bottleneck?
    """
    def should_use_mp(self, i):
        return type(i) == slice

    def should_use_mp(self, i):
        return type(i) == slice or type(i) == list

    def getItemSlice(self, i):
        #assign index as placeholder for result in toReturn
        if type(i) == slice:
            placeholder = [j for j in range(*i.indices(len(self)))] #use to look up correct index because using the ".index" method in an array holding arrays leads to comparison error
            toReturn = [j for j in range(*i.indices(len(self)))]
        elif type(i) == list: #indexing by list
            placeholder = [j for j in i]
            toReturn = [j for j in i]
        if hasattr(self, "use_mp") and self.use_mp == False: #in case it makes more sense to just use a loop instead of dealing with overhead of starting processes
            for i, j in enumerate(toReturn):
                toReturn[i] = self[j]
            return toReturn
        manager = mp.Manager()
        inQ = manager.Queue()
        outQ = manager.Queue()
        if self.n_process > 1: #otherwise use for loop
            [inQ.put(j) for j in toReturn]
            [inQ.put(None) for j in range(self.n_process)]
            processes = [
                mp.Process(
                    target=self.helper_process,
                    args=(
                        inQ,
                        outQ)) for j in range(
                    self.n_process)]
            if not hasattr(self, "verbose") or self.verbose == True:
                print("Starting {} processes".format(self.n_process))
            [p.start() for p in processes]
            [p.join() for p in processes]
            startIndex = toReturn[0]
        while not outQ.empty():
            place, res = outQ.get()
            index = placeholder.index(place)
            if type(res) == int:
                if not hasattr(self, "verbose") or self.verbose == True:
                    print("SLURM sent OOM event, retrying: ", res)
                res = self[place] #slurm sent oom event, we gotta try again.
            toReturn[index] = res
        for index, res in enumerate(toReturn):
            if type(res) == int:
                toReturn[index] = self[res]
        return toReturn
        # return Pool().map(self.__getitem__, toReturn)

    def helper_process(self, in_q, out_q):
        for i in iter(in_q.get, None):
            if not hasattr(self, "verbose") or self.verbose == True:
                if not hasattr(self, "verbosity"):
                    self.verbosity = 250
                if i % self.verbosity == 0:
                    print("retrieving: {}".format(i))
            out_q.put((i, self[i]))

class CoherenceTransformer(MultiProcessingDataset):
    def __init__(self, edfRawData, n_process=None, coherence_all=True, coherence_pairs=None, average_coherence=True, coherence_bin=None, columns_to_use=[], is_pandas=True, is_tuple_data=True):
        """
        Parameters
        ----------
        edfRawData : DataFrame
            An array-like holding the data for coherence
        n_process : int
            number of processes to use when indexing a slice
        coherence_all : bool
            If to do pair-wise coherence on all channels, if so we increase
            num features to n*n-1
        coherence_pairs : list
            If coherence_all is false, pass in a list of tuples holding columns
            to run coherence measurements on
        average_coherence : bool
            If true, just do an average of all coherences over all represented
            frequencies. If False, use coherence_bin to histogram bin everything
        Returns
        -------
        CoherenceTransformer
            Array-like
        """
        self.edfRawData = edfRawData
        self.n_process = n_process
        self.is_pandas = is_pandas
        self.coherence_all = coherence_all
        self.coherence_pairs = coherence_pairs
        self.average_coherence = average_coherence
        self.coherence_bin = coherence_bin
        self.columns_to_use = columns_to_use
        self.is_pandas = is_pandas
        self.is_tuple_data = is_tuple_data
    def __len__(self):
        return len(self.edfRawData)
    def __getitem__(self, i):
        if self.should_use_mp(i):
            print("starting simple coherence")
            return self.getItemSlice(i)
        if self.is_tuple_data:
            raw_data, ann = self.edfRawData[i]
        else:
            raw_data = self.edfRawData[i]
        if self.coherence_all:
            coherence_pairs = []
            if self.is_pandas:
                for k in range(len(self.columns_to_use) - 1):
                    column_1 = self.columns_to_use[k]
                    for j in range(k + 1, len(self.columns_to_use)):
                        column_2 = self.columns_to_use[j]
                        coherence_pairs.append((column_1, column_2))
            else:
                for k in range(raw_data.shape[1] - 1):
                    for j in range(k + 1, raw_data.shape[1]):
                        coherence_pairs.append((j, k))

        else:
            coherence_pairs = self.coherence_pairs

        if self.average_coherence:
            toReturn = pd.Series()
            if self.is_pandas:
                for column_1, column_2 in coherence_pairs:
                    toReturn["coherence {}".format((column_1, column_2))] =  np.mean(coherence(raw_data[column_1], raw_data[column_2], fs=constants.COMMON_FREQ, nperseg=constants.COMMON_FREQ/4)[1])
            else:
                for column_1, column_2 in coherence_pairs:
                    toReturn["coherence {}".format((column_1, column_2))] =  np.mean(coherence(raw_data.T[column_1], raw_data.T[column_2], fs=constants.COMMON_FREQ, nperseg=constants.COMMON_FREQ/4)[1])
        else:
            # raise Exception("Not implemented yet")

            if self.is_pandas:
                raise Exception("Not implemented yet")
            else:
                toReturn = pd.Series()
                window_count_size = int(
                    self.coherence_bin /
                    pd.Timedelta(
                        seconds=constants.COMMON_DELTA))

                original_data_label = self.edfRawData[i]
                if self.is_tuple_data:
                    original_data, ann = original_data_label
                else:
                    original_data = original_data_label
                if self.is_pandas:
                    original_data = original_data.values
                else:
                    original_data = original_data
                coher_data = np.ndarray((int(original_data.shape[0]/window_count_size), int(original_data.shape[1] * (original_data.shape[1] - 1) / 2) ))
                for i in range(coher_data.shape[0]):
                    start_time = i * window_count_size
                    end_time = (i+1) * window_count_size
                    for j, coher_pair in enumerate(coherence_pairs):
                        coher_data[i, j] = np.mean(coherence(raw_data[start_time:end_time, coher_pair[0], ], raw_data[start_time:end_time, coher_pair[1], ], fs=constants.COMMON_FREQ, nperseg=constants.COMMON_FREQ/4)[1])
                toReturn = coher_data
        return toReturn, ann


def np_rolling_window(a, window):
    # https://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


class Flattener(MultiProcessingDataset):
    def __init__(self, dataset, is_tuple_data=True, is_pandas_data=True, n_process=4):
        self.dataset = dataset
        self.is_tuple_data = is_tuple_data
        self.is_pandas_data = is_pandas_data
        self.n_process = n_process

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        if self.should_use_mp(i):
            return self.getItemSlice(i)
        else:
            if self.is_tuple_data:
                data, label = self.dataset[i]
            else:
                data = self.dataset[i]

            if self.is_pandas_data:
                data = data.values
            data = data.flatten()
            if self.is_tuple_data:
                return data, label
            else:
                return data


class EdfFFTDatasetTransformer(MultiProcessingDataset):
    freq_bins = [0.2 * i for i in range(50)] + list(range(10, 80, 1)) #default freq bins unless if you override this
    """Implements an indexable dataset applying fft to entire timeseries,
        returning histogram bins of fft frequencies
    Parameters
    ----------
    edf_dataset : EdfDataset
        an array-like returning the raw channel data and the output as a tuple or a single result
    freq_bins : type
        Description of parameter `freq_bins`.
    n_process : type
        Description of parameter `n_process`.
    precache : type
        Description of parameter `precache`.
    window_size : type
        Description of parameter `window_size`.
    non_overlapping : type
        Description of parameter `non_overlapping`.
    """

    def __init__(
        self,
        edf_dataset,
        is_tuple_data=True,
        is_pandas_data=True,
        freq_bins=freq_bins,
        n_process=None,
        precache=False,
        window_size=None,
        non_overlapping=True,
        return_ann=True,
        return_numpy=False #return pandas.dataframe if possible (if windows_size is false)
    ):
        """Used to read the raw data in
        Parameters
        ----------
        edf_dataset : EdfDataset
            Array-like returning the channel data (channel by time) and annotations (doesn't matter what the shape is)
        freq_bins : array
            Used to segment the frequencies into histogram bins
        n_process : int
            Used to define the number of processes to use for large reads in. If None, uses cpu count
        precache : bool
            Use to load all data at beginning and keep cache of it during operations
        window_size : pd.Timedelta
            If None, runs the FFT on the entire datset. If set, uses overlapping windows to run fft on
        non_overlapping : bool
            If true, the windows are used to reduce dim red, we don't use rolling-like behavior
        return_ann : bool
            If false, we just output the raw data
        Returns
        -------
        None
        """
        self.return_numpy = return_numpy
        if not is_tuple_data:
            return_ann = False #you can't return annotation data if annotation isn't included
        self.is_tuple_data = is_tuple_data
        self.is_pandas_data = is_pandas_data
        self.edf_dataset = edf_dataset
        if n_process is None:
            n_process = mp.cpu_count()
        self.n_process = n_process
        self.precache = False
        self.freq_bins = freq_bins
        self.window_size = window_size
        self.non_overlapping = non_overlapping
        self.return_ann = return_ann
        if precache:
            print(
                "starting precache job with: {} processes".format(
                    self.n_process))
            self.data = self[:]
        self.precache = precache

    def __len__(self):
        return len(self.edf_dataset)

    def __getitem__(self, i):
        if self.precache:
            return self.data[i]
        if self.should_use_mp(i):
            return self.getItemSlice(i)
        if self.window_size is None:
            original_data_label = self.edf_dataset[i]
            if self.is_tuple_data:
                original_data, label = original_data_label
            else:
                original_data = original_data_label
            if self.is_pandas_data:
                columns = original_data.columns
                original_data = original_data.values
            else:
                columns = list(range(original_data.shape[1]))
            fft_data = np.nan_to_num(
                np.abs(
                    np.fft.fft(
                        original_data,
                        axis=0)))
            fft_freq = np.fft.fftfreq(fft_data.shape[0], d=constants.COMMON_DELTA)
            fft_freq_bins = self.freq_bins
            new_fft_hist = pd.DataFrame(
                index=fft_freq_bins[:-1], columns=columns)
            for i, name in enumerate(columns):
                new_fft_hist[name] = np.histogram(
                    fft_freq, bins=fft_freq_bins, weights=fft_data[:, i])[0]
            if self.return_numpy:
                new_fft_hist = new_fft_hist.values
            if not self.return_ann:
                return new_fft_hist
            return new_fft_hist, label
        else:
            window_count_size = int(
                self.window_size /
                pd.Timedelta(
                    seconds=constants.COMMON_DELTA))

            original_data_label = self.edf_dataset[i]
            if self.is_tuple_data:
                original_data, label = original_data_label
            else:
                original_data = original_data_label
            if self.is_pandas_data:
                fft_data = original_data.values
            else:
                fft_data = original_data
            fft_data_windows = np_rolling_window(
                np.array(fft_data.T), window_count_size)
            if self.non_overlapping:
                fft_data_windows = fft_data_windows[:, list(
                    range(0, fft_data_windows.shape[1], window_count_size))]
            fft_data = np.abs(
                np.fft.fft(
                    fft_data_windows,
                    axis=2))  # channel, window num, frequencies
            fft_freq_bins = self.freq_bins
            new_hist_bins = np.zeros(
                (fft_data.shape[0], fft_data.shape[1], len(fft_freq_bins) - 1))
            fft_freq = np.fft.fftfreq(window_count_size, d=constants.COMMON_DELTA)
            for i, channel in enumerate(fft_data):
                for j, window_channel in enumerate(channel):
                    new_hist_bins[i, j, :] = np.histogram(
                        fft_freq, bins=fft_freq_bins, weights=window_channel)[0]
            if not self.return_ann:
                return new_hist_bins
            if (hasattr(self.edf_dataset, "expand_tse") and self.edf_dataset.expand_tse and not self.non_overlapping):
                return new_hist_bins, label.rolling(window_count_size).mean(
                )[:-window_count_size + 1].fillna(method="ffill").fillna(method="bfill")
            elif (hasattr(self.edf_dataset, "expand_tse") and self.edf_dataset.expand_tse and self.non_overlapping):
                annotations = label.rolling(window_count_size).mean()[
                    :-window_count_size + 1]
                return new_hist_bins, annotations.iloc[list(range(
                    0, annotations.shape[0], window_count_size))].fillna(method="ffill").fillna(method="bfill")
            else:
                return new_hist_bins, label
            
class SimpleHandEngineeredDataset(MultiProcessingDataset):
    def __init__(self, edfRawData, n_process=None, features = [], f_names = [], max_size=None, vectorize=None, is_pandas_data=True):
        assert len(features) == len(f_names)
        self.edfRawData = edfRawData
        self.n_process = n_process
        if n_process is None:
            self.n_process = mp.cpu_count()
        self.features = features
        self.f_names = f_names
        self.max_size = max_size
        self.vectorize = vectorize
        self.is_pandas_data = is_pandas_data

    def __len__(self):
        return len(self.edfRawData)

    def __getitem__(self, i):
        if self.should_use_mp(i):
            print("starting simple handEngineeredData")
            return self.getItemSlice(i)
        rawData, ann = self.edfRawData[i]
        if self.max_size is not None and max(rawData.index) < self.max_size:
            rawData = rawData[:self.max_size]
        if not self.is_pandas_data:
            rawData = pd.DataFrame(rawData)
        handEngineeredData = pd.DataFrame(index=rawData.columns, columns=self.f_names)

        for i, feature in enumerate(self.features):
            handEngineeredData[self.f_names[i]] = rawData.apply(lambda x: feature(x.values))
        if self.vectorize == "full":
            return handEngineeredData.values.reshape(-1)
        if self.vectorize == "mean":
            return handEngineeredData.values.mean()
        return handEngineeredData

class EdfDWTDatasetTransformer(MultiProcessingDataset):
    def __init__(
        self,
        edf_dataset,
        n_process=None,
        precache=False,
        wavelet="db1",
        return_ann=True,
        max_coef=None,
    ):
        """Used to read the raw data in
        Parameters
        ----------
        edf_dataset : EdfDataset
            Array-like returning the channel data (channel by time) and annotations (doesn't matter what the shape is)
        freq_bins : array
            Used to segment the frequencies into histogram bins
        n_process : int
            Used to define the number of processes to use for large reads in. If None, uses cpu count
        precache : bool
            Use to load all data at beginning and keep cache of it during operations
        window_size : pd.Timedelta
            If None, runs the FFT on the entire datset. If set, uses overlapping windows to run fft on
        non_overlapping : bool
            If true, the windows are used to reduce dim red, we don't use rolling-like behavior
        return_ann : bool
            If false, we just output the raw data
        Returns
        -------
        None
        """
        self.edf_dataset = edf_dataset
        if n_process is None:
            n_process = mp.cpu_count()
        self.n_process = n_process
        self.precache = False
        self.return_ann = return_ann
        if precache:
            print(
                "starting precache job with: {} processes".format(
                    self.n_process))
            self.data = self[:]
        self.precache = precache
        self.wavelet = wavelet
        self.max_coef = max_coef

    def __len__(self):
        return len(self.edf_dataset)

    def __getitem__(self, i):
        if self.precache:
            return self.data[i]
        if self.should_use_mp(i):
            return self.getItemSlice(i)
        original_data = self.edf_dataset[i]
        return original_data.apply(
            lambda x: pywt.dwt(
                x.values,
                self.wavelet)[0],
            axis=0)[
            :self.max_coef]