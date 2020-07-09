import sys, os
sys.path.append(os.path.realpath(".."))
import tsfresh.feature_extraction.feature_calculators as tsf
from sacred.observers import MongoObserver
from sacred import SETTINGS
SETTINGS["CONFIG"]["READ_ONLY_CONFIG"] = False #weird issue where GridSearchCV alters one of the config values
import pickle as pkl
from addict import Dict
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from os import path
import sys
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer, accuracy_score, roc_auc_score, matthews_corrcoef, classification_report, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle as pkl
import sacred
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
ex = sacred.Experiment(name="seizure_predict_baseline_traditional_ml")


import util_funcs
ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))


@ex.named_config
def rf():
    parameters = {
        'rf__criterion': ["gini", "entropy"],
        'rf__n_estimators': [ 400, 600, 1200],
#         'rf__n_estimators': [50,  ],
        'rf__max_features': ['auto', 'log2', 30],
        'rf__max_depth': [None, 2, 8], #smaller max depth, gradient boosting, more max features
        'rf__min_samples_split': [2, 4, 8],
        'rf__n_jobs': [1],
        'rf__min_weight_fraction_leaf': [0, 0.2, 0.5],
        # 'imb__method': [None, util_funcs.ImbalancedClassResampler.SMOTE, util_funcs.ImbalancedClassResampler.RANDOM_UNDERSAMPLE]
    }
    clf_name = "rf"
    clf_step = ('rf', RandomForestClassifier())

@ex.named_config
def svc():
    parameters = {
        'svc__C':[1, 0.5, 1.5, 2, 4,8],
        'svc__kernel':['linear', 'rbf', "poly", 'sigmoid'],
        'svc__gamma':[1,2,3,5,10,20, 'auto', 'scale'],
        'svc__shrinking': [True, False],
        'svc__probability': [True, False],
        'svc__max_iter': [1000]
    }
    # parameters.append(parameters[0].copy())
    # parameters[1]["kernel"] = "poly"
    # parameters[1]["degree"] = [1,2,3,4,6,8,12,16]
    clf_name = "svc"
    clf_step = ('svc', SVC())

@ex.named_config
def xgboost():
    parameters = {
        "xgboost__max_depth": [2,3,4,5,6,10,12],
        "xgboost__learning_rate":[0.1,0.2],
        "xgboost__gamma":[0,0.1,0.2],
        "xgboost__reg_alpha":[0,0.1,0.2],
        "xgboost__reg_lambda":[0,0.1,0.2,1,2,5],
        "xgboost__n_estimators":[100,200,300,400,600],
    }
    clf_name = "xgboost"
    clf_step = (clf_name, xgb.XGBRegressor(objective='binary:logistic',))
    use_xgboost=True

@ex.named_config
def rf_debug():
    parameters = {'rf__criterion': ["gini", "entropy"],}

@ex.named_config
def lr():
    parameters = {
        'lr__tol': [0.001, 0.0001, 0.00001],
        'lr__multi_class': ["multinomial"],
        'lr__C': [0.05, .1, .2, .4, .8],
        'lr__solver': ["sag"],
        'lr__max_iter': [1000],
        'lr__n_jobs': [1]
    }
    clf_name = "lr"
    clf_step = ('lr', LogisticRegression())

@ex.named_config
def debug():
    max_samples=1000
    max_bckg_samps_per_file=20


@ex.config
def config():
    train_split = "train"
    use_random_cv = False
    test_split = "dev_test"
    ref = "01_tcp_ar"
    include_simple_coherence = True
    parameters = {}
    clf_step = None
    clf_name = ''
    num_files = None
    freq_bins=[0,3.5,7.5,14,20,25,40]
    n_process = 7
    precache = True
    train_pkl="/datadrive/TUH_EEG/mups-shallow-features/extracted_train_features_v2.pkl"
    valid_pkl="/datadrive/TUH_EEG/mups-shallow-features/extracted_val_features_v2.pkl"
    test_pkl= "/datadrive/TUH_EEG/mups-shallow-features/extracted_test_features_v2.pkl"
    max_bckg_samps_per_file = 100
    max_samples=None
    regenerate_data=False
    imbalanced_resampler = "rul"
    pre_cooldown=4
    post_cooldown=None
    sample_time=4
    num_seconds=1
    use_xgboost = False
    use_simple_hand_engineered_features=True
    random_under_sample_data_gen = False


@ex.capture
def get_data(mode, max_samples, n_process, complex_feature_channels, max_bckg_samps_per_file,use_simple_hand_engineered_features, random_under_sample_data_gen, num_seconds, ref="01_tcp_ar", num_files=None, freq_bins=[0,3.5,7.5,14,20,25,40],  include_simple_coherence=True,):
    eds = getDataSampleGenerator()
    train_label_files_segs = eds.get_train_split()
    test_label_files_segs = eds.get_test_split()
    valid_label_files_segs = eds.get_valid_split()

    #increased n_process to deal with io processing
    train_edss = er.EdfDatasetSegmentedSampler(segment_file_tuples=train_label_files_segs, mode=mode, random_under_sample=random_under_sample_data_gen, num_samples=max_samples, max_bckg_samps_per_file=max_bckg_samps_per_file, n_process=int(n_process*2), gap=num_seconds*pd.Timedelta(seconds=1))[:]
    valid_edss = er.EdfDatasetSegmentedSampler(segment_file_tuples=valid_label_files_segs, mode=mode, random_under_sample=random_under_sample_data_gen, num_samples=max_samples, max_bckg_samps_per_file=max_bckg_samps_per_file, n_process=int(n_process*2), gap=num_seconds*pd.Timedelta(seconds=1))[:]
    test_edss = er.EdfDatasetSegmentedSampler(segment_file_tuples=test_label_files_segs, mode=mode, random_under_sample=random_under_sample_data_gen, num_samples=max_samples, max_bckg_samps_per_file=max_bckg_samps_per_file, n_process=int(n_process*2), gap=num_seconds*pd.Timedelta(seconds=1))[:]
    def simple_edss(edss):
        '''
        Use only a few columns so that we don't make 21*20 coherence pairs
        '''
        all_channels = util_funcs.get_common_channel_names()
        subset_channels = [all_channels.index(channel) for channel in complex_feature_channels]
        return [(datum[0][:, subset_channels], datum[1]) for datum in edss]
    if include_simple_coherence:
        trainCoherData = np.stack([datum.values for datum in [datum[0] for datum in wfdata.CoherenceTransformer(simple_edss(train_edss), columns_to_use=constants.SYMMETRIC_COLUMN_SUBSET, n_process=n_process, is_pandas=False)[:]]])
        validCoherData = np.stack([datum.values for datum in [datum[0] for datum in wfdata.CoherenceTransformer(simple_edss(valid_edss), columns_to_use=constants.SYMMETRIC_COLUMN_SUBSET, n_process=n_process, is_pandas=False)[:]]])
        testCoherData = np.stack([datum.values for datum in  [datum[0] for datum in wfdata.CoherenceTransformer(simple_edss(test_edss), columns_to_use=constants.SYMMETRIC_COLUMN_SUBSET, n_process=n_process, is_pandas=False)[:]]])
    if use_simple_hand_engineered_features:
        trainSHED = wfdata.SimpleHandEngineeredDataset(simple_edss(train_edss), n_process=n_process, is_pandas_data=False, features=[tsf.abs_energy, tsf.sample_entropy, lambda x: tsf.number_cwt_peaks(x, int(constants.COMMON_FREQ/25))], f_names=["abs_energy", "entropy", "num_peaks"], vectorize="full")[:]
        validSHED = wfdata.SimpleHandEngineeredDataset(simple_edss(valid_edss), n_process=n_process, is_pandas_data=False, features=[tsf.abs_energy, tsf.sample_entropy, lambda x: tsf.number_cwt_peaks(x, int(constants.COMMON_FREQ/25))], f_names=["abs_energy", "entropy", "num_peaks"], vectorize="full")[:]
        testSHED = wfdata.SimpleHandEngineeredDataset(simple_edss(test_edss), n_process=n_process, is_pandas_data=False, features=[tsf.abs_energy, tsf.sample_entropy, lambda x: tsf.number_cwt_peaks(x, int(constants.COMMON_FREQ/25))], f_names=["abs_energy", "entropy", "num_peaks"], vectorize="full")[:]

    train_edss = read.Flattener(read.EdfFFTDatasetTransformer(train_edss, freq_bins=freq_bins, is_pandas_data=False), n_process=n_process)[:]
    valid_edss = read.Flattener(read.EdfFFTDatasetTransformer(valid_edss, freq_bins=freq_bins, is_pandas_data=False), n_process=n_process)[:]
    test_edss = read.Flattener(read.EdfFFTDatasetTransformer(test_edss, freq_bins=freq_bins, is_pandas_data=False), n_process=n_process)[:]
    def split_tuples(data):
        return np.stack([datum[0] for datum in data]), np.stack([datum[1] for datum in data])
    train_edss, train_labels = split_tuples(train_edss)
    valid_edss, valid_labels = split_tuples(valid_edss)
    test_edss, test_labels = split_tuples(test_edss)


    if include_simple_coherence:
        train_edss = np.hstack([train_edss, trainCoherData])
        valid_edss = np.hstack([valid_edss, validCoherData])
        test_edss = np.hstack([test_edss, testCoherData])

    if use_simple_hand_engineered_features:
        train_edss = np.hstack([train_edss, np.stack(trainSHED)])
        valid_edss = np.hstack([valid_edss, np.stack(validSHED)])
        test_edss = np.hstack([test_edss, np.stack(testSHED)])


    print("Data Shape:", train_edss.shape)

    #some of the features are returning nans (assuming there is a log that may not play well?)
    return (np.nan_to_num(train_edss), train_labels), \
        (np.nan_to_num(valid_edss), valid_labels), \
        (np.nan_to_num(test_edss), test_labels)

@ex.capture
def getImbResampler(imbalanced_resampler):
    if imbalanced_resampler is None:
        return None
    elif imbalanced_resampler == "SMOTE":
        return SMOTE()
    elif imbalanced_resampler == "rul":
        return RandomUnderSampler()

@ex.capture
def resample_x_y(x, y, imbalanced_resampler):
    if imbalanced_resampler is None:
        return x, y
    else:
        return getImbResampler().fit_resample(x, y)

@ex.capture
def getGridsearch(valid_indices, clf_step, parameters, n_process, use_random_cv, num_random_choices=10, use_xgboost=False):
    steps = [
        # ("imb", util_funcs.ImbalancedClassResampler(n_process=n_process)),
        clf_step
    ]



    pipeline = Pipeline(steps)
    if use_xgboost:
        scorer = make_scorer(roc_auc_score)
    else:
        scorer = make_scorer(f1_score, average="macro")
    if use_random_cv:
        return RandomizedSearchCV(pipeline, parameters, cv=valid_indices,
                            scoring=scorer, n_jobs=n_process, n_iter=num_random_choices)
    return GridSearchCV(pipeline, Dict(parameters), cv=valid_indices,
                        scoring=scorer, n_jobs=n_process)


@ex.capture
def getFeatureScores(gridsearch, clf_name):
    if clf_name == "lr":
        return gridsearch.best_estimator_.named_steps[clf_name].coef_
    elif clf_name == "rf":
        return gridsearch.best_estimator_.named_steps[clf_name].feature_importances_
    elif clf_name == "xgboost":
        return gridsearch.best_estimator_.named_steps[clf_name].feature_importances_


@ex.main
def main(train_pkl, valid_pkl, test_pkl, train_split, num_seconds, imbalanced_resampler, test_split, clf_name, precache, regenerate_data, use_xgboost, num_random_choices=10):
    
    trainXY = pkl.load(open(train_pkl, "rb"))
    validXY = pkl.load(open(valid_pkl, "rb"))
    testXY = pkl.load(open(test_pkl, "rb"))
    
    trainDataResampled = np.array([datum[0] for datum in trainXY])
    validDataResampled = np.array([datum[0] for datum in validXY])
    trainLabelsResampled = np.array([datum[1] for datum in trainXY])
    validLabelsResampled = np.array([datum[1] for datum in validXY])
    testData = np.array([datum[0] for datum in testXY])
    testLabels = np.array([datum[1] for datum in testXY])




    print("Starting ", clf_name)



    # if use_xgboost:
    #     return xgboost_flow(trainDataResampled, trainLabelsResampled, validDataResampled, validLabelsResampled, testData, testLabels)

    trainValidData = np.vstack([trainDataResampled, validDataResampled])
    trainValidLabels = np.hstack([trainLabelsResampled, validLabelsResampled])
    trainValidData = trainValidData.astype(np.float32)
    trainValidData[trainValidData == np.inf] = 0
    testData = testData.astype(np.float32)
    testData[testData == np.inf] = 0

    trainValidData = np.nan_to_num(trainValidData.astype(np.float32))
    testData = np.nan_to_num(testData.astype(np.float32))

    valid_indices = [[[i for i in range(len(trainLabelsResampled))], [i + len(trainLabelsResampled) for i in range(len(validLabelsResampled))]]]
    gridsearch = getGridsearch(valid_indices)

    gridsearch.fit(trainValidData, trainValidLabels)
    print(pd.Series(trainLabelsResampled).value_counts())

    print("Best Parameters were: ", gridsearch.best_params_)
    print(pd.Series(testLabels).value_counts())

    bestPredictor = gridsearch.best_estimator_.named_steps[clf_name]
    bestPredictor.fit(trainValidData, trainValidLabels)
    y_pred = bestPredictor.predict(testData)

    if y_pred.dtype == np.float32 or y_pred.dtype == np.float:
        y_pred = y_pred > 0.5

#     print("Proportion True in Predicted Test Set: ", y_pred.sum() / len(testLabels),
#           "Proportion False in Predicted Test Set: ", 1 - y_pred.sum() / len(testLabels))

#     print("F1_score: ", f1_score(y_pred, testLabels))
#     print("accuracy: ", accuracy_score(y_pred, testLabels))
#     print("MCC: ", matthews_corrcoef(y_pred, testLabels))
    try:
        auc = roc_auc_score(y_pred, testLabels)
        print("AUC: ", auc)
    except Exception:
        auc = "cannot be calculated"

    # print("auc: ", auc(y_pred, testGenders))
    toSaveDict = Dict()
    toSaveDict.getFeatureScores = getFeatureScores(gridsearch)
    toSaveDict.gridsearch = gridsearch
    toSaveDict.best_params_ = gridsearch.best_params_

    fn = "seizure{}_{}_{}.pkl".format(clf_name, num_seconds, imbalanced_resampler if imbalanced_resampler is not None else "noresample")
    pkl.dump(toSaveDict, open(fn, 'wb'))
    ex.add_artifact(fn)

    return {'test_scores': {
        'f1': {"macro": f1_score(y_pred, testLabels, average="macro"), "weighted": f1_score(y_pred, testLabels, average="weighted"), "micro": f1_score(y_pred, testLabels, average="micro")},
        'acc': accuracy_score(y_pred, testLabels),
        'auc': auc,
        "classification_report": classification_report(testLabels, y_pred, output_dict=True),
        "best_params":  gridsearch.best_params_
    }}


if __name__ == "__main__":
    #https://github.com/ContinuumIO/anaconda-issues/issues/11294#issuecomment-533138984 mp got jacked by version upgrade

    ex.run_commandline()
