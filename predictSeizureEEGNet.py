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

from addict import Dict
from tensorflow.keras.optimizers import Adam
ex = sacred.Experiment(name="seizure_baseline_eegnet_etc")
ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))

@ex.capture
def trainDataSource(train_val_hd5_location):
    with h5py.File(train_val_hd5_location) as trainFile:
        train_x = trainFile['train_x'].value
        train_x = train_x.transpose(0,2,1).reshape((-1,1,22,1000))
        train_y = trainFile['train_y'].value
        return train_x, to_categorical(train_y, 4)
    
@ex.capture
def validDataSource(train_val_hd5_location):
    with h5py.File(train_val_hd5_location) as trainFile:
        train_x = trainFile['val_x'].value
        train_x = train_x.transpose(0,2,1).reshape((-1,1,22,1000))
        train_y = trainFile['val_y'].value
        return train_x, to_categorical(train_y, 4)
    
@ex.capture
def testDataSource(test_hd5_location):
    with h5py.File(test_hd5_location) as testFile:
        x = testFile['test_x'].value
        x = x.transpose(0,2,1).reshape((-1,1,22,1000))
        y = testFile['test_y'].value
        return x, to_categorical(y, 4)
    
def randomString(stringLength=16):
    """Generate a random string of fixed length """
    letters = string.ascii_uppercase
    return ''.join(random.choice(letters) for i in range(stringLength))

@ex.config
def config():
    num_channels = 22
    time_steps = 1000
    nb_classes = 4
    train_val_hd5_location= "/datadrive/TUH_EEG/MUPS/data/cross_sub_TUH_EEG/cross_subject_data_train_val.hdf5"
    test_hd5_location = "/datadrive/TUH_EEG/MUPS/data/cross_sub_TUH_EEG/cross_subject_data_test.hdf5"
    model_type = "eegnet"
    lr = 0.0001
    patience = 20
    num_epochs = 200
    batch_size=16
    early_stopping_on = "val_loss"
    model_name = "/home/mohammed/" + randomString() + ".h5" #set to rando string so we don't have to worry about collisions

@ex.named_config
def debug():
    patience = 1
    num_epochs = 1
    
@ex.capture
def return_model(model_type, nb_classes, num_channels, time_steps):
    if model_type == "eegnet":
        return EEGNet(nb_classes = nb_classes, Chans = num_channels, Samples = time_steps)
    elif model_type == "shallow_eegnet":
        return ShallowConvNet(nb_classes = nb_classes, Chans = num_channels, Samples = time_steps)
    elif model_type == "deep_eegnet":
        return DeepConvNet(nb_classes = nb_classes, Chans = num_channels, Samples = time_steps)
    else:
        raise Exception(f"bad model type: {model_type}")
        
@ex.capture
def return_compiled_model(lr):
    model = return_model()
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=lr))
    model.summary()
    return model

@ex.capture
def get_model_checkpoint(model_name, early_stopping_on):
    return ModelCheckpoint(model_name, monitor=early_stopping_on, save_best_only=True, verbose=1)


@ex.capture
def get_early_stopping(patience, early_stopping_on):
    return EarlyStopping(patience=patience, verbose=1, monitor=early_stopping_on)

@ex.capture
def get_cb_list():
    return [get_model_checkpoint(), get_early_stopping()]
    
@ex.capture
def load_best_model(model_name):
    return keras.models.load_model(model_name)

@ex.main
def main(batch_size, num_epochs):
    keras.backend.set_image_data_format("channels_first")
    model = return_compiled_model()
    train_x, train_y = trainDataSource()
    valid_x, valid_y = validDataSource()
    history = model.fit(train_x, train_y, batch_size=batch_size, epochs=num_epochs, validation_data=([valid_x], valid_y), callbacks=get_cb_list())
    history = history.history
    results = Dict()
    results.history = history
    test_x, test_y = testDataSource()
    
    model = load_best_model()
    val_predictions = model.predict(valid_x, batch_size=64)

    results.valid.classification_report = classification_report(valid_y.argmax(1), val_predictions.argmax(1), output_dict=True)
    results.valid.confustion_matrix = confusion_matrix(valid_y.argmax(1), val_predictions.argmax(1))
    results.valid.f1_score.macro =  f1_score(valid_y.argmax(1), val_predictions.argmax(1), average="macro")
    results.valid.f1_score.micro =  f1_score(valid_y.argmax(1), val_predictions.argmax(1), average="micro")
    results.valid.f1_score.weighted =  f1_score(valid_y.argmax(1), val_predictions.argmax(1), average="weighted")
    


    train_predictions = model.predict(train_x, batch_size=64)
    results.train.classification_report = classification_report(train_y.argmax(1), train_predictions.argmax(1), output_dict=True)
    results.train.confustion_matrix = confusion_matrix(train_y.argmax(1), train_predictions.argmax(1))
    results.train.f1_score.macro =  f1_score(train_y.argmax(1), train_predictions.argmax(1), average="macro")
    results.train.f1_score.micro =  f1_score(train_y.argmax(1), train_predictions.argmax(1), average="micro")
    results.train.f1_score.weighted =  f1_score(train_y.argmax(1), train_predictions.argmax(1), average="weighted")
    
    test_predictions = model.predict(test_x, batch_size=64)
    results.test.classification_report = classification_report(test_y.argmax(1), test_predictions.argmax(1), output_dict=True)
    results.test.confustion_matrix = confusion_matrix(test_y.argmax(1), test_predictions.argmax(1))
    results.test.f1_score.macro =  f1_score(test_y.argmax(1), test_predictions.argmax(1), average="macro")
    results.test.f1_score.micro =  f1_score(test_y.argmax(1), test_predictions.argmax(1), average="micro")
    results.test.f1_score.weighted =  f1_score(test_y.argmax(1), test_predictions.argmax(1), average="weighted")
    return results.to_dict()


if __name__ == "__main__":
    ex.run_commandline()