import sys, os
sys.path.append(os.path.realpath(".."))
from sacred.observers import MongoObserver
import pickle as pkl
from addict import Dict
import pandas as pd
import numpy as np
import numpy.random as random
from os import path
from keras import backend as K

# from multiprocessing import Process
import constants
import util_funcs
import functools
from sklearn.metrics import f1_score, make_scorer, accuracy_score, roc_auc_score, matthews_corrcoef, classification_report, log_loss, confusion_matrix, mean_squared_error

import sacred
import keras

import random
import string
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.utils import multi_gpu_model
from time import time

from addict import Dict
ex = sacred.Experiment(name="seizure_baseline_eegnet_etc")

@ex.capture
def dataSource():
    

@ex.config
def config():
    num_channels = 21
    time_steps = 400
    train_val_hd5_location= "/datadrive/TUH_EEG/MUPS/data/cross_sub_TUH_EEG/cross_subject_data_train_val.hdf5"
    test_hd5_location = "/datadrive/TUH_EEG/MUPS/data/cross_sub_TUH_EEG/cross_subject_data_test.hdf5"
    model_name = "eeg_net_original
    
@ex.capture
def return_model(model_name, num_channels, time_steps):
    
@ex.main
def main():
    print("hello world")
    edg, valid_edg, test_edg, len_all_patients = get_data_generators()
    model = lstm_model()


if __name__ == "__main__":
    ex_lstm.run_commandline()