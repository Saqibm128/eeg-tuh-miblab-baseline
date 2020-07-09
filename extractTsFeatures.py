import sys, os
sys.path.append(os.path.realpath(".."))
from sacred.observers import MongoObserver
import pickle as pkl
from addict import Dict
import pandas as pd
import numpy as np
import numpy.random as random
from os import path

# from multiprocessing import Process
import functools
from sklearn.metrics import f1_score, make_scorer, accuracy_score, roc_auc_score, matthews_corrcoef, classification_report, log_loss, confusion_matrix, mean_squared_error

import sacred
from sacred.observers import MongoObserver
import util_funcs
import random
import string
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

from time import time
import h5py
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical

from eegmodels import EEGNet, ShallowConvNet, DeepConvNet
from dataGen import EdfDataGenerator, RandomRearrangeBatchGenerator
from addict import Dict
from tensorflow.keras.optimizers import Adam
from predictSeizureEEGNet import ex as eeg_ingredient
ex = sacred.Experiment(name="extract_shallow_learner_features", ingredients=[eeg_ingredient])
ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))


@ex.capture
def extractAllFeatures():
    

@ex.main
def main():

if __name__ == "__main__":
    ex.run_commandline()