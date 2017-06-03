import sys

sys.path.insert(1, '../Core-scripts')

from parse_and_prepare import ProteinProteinInteractionClassifier as ppi
import file_readers as fr
import prediction as pred
import pickle
import os
import re
import numpy as np
from gensim.models import word2vec
import logging
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sys import argv
import multiprocessing
import pprint
import numpy as np
import operator

random_seeds = [144, 235, 905, 2895, 3462, 4225, 5056, 5192, 7751, 7813]

if not os.path.isdir('Results/train_test/'):
    os.makedirs('Results/train_test/')
if not os.path.isdir('Results/train_val/'):
    os.makedirs('Results/train_val/')
if not os.path.isdir('Results/models/'):
    os.makedirs('Results/models/')
if not os.path.isdir('Results/result_list/'):
    os.makedirs('Results/result_list/')
if not os.path.isdir('Results/metrics/'):
    os.makedirs('Results/metrics/')

def open_files():
    strict = pickle.load(open('Results/drosophila_mentions_strict_real.pkl', 'rb'))
    gen = pickle.load(open('Results/drosophila_mentions_gen_real.pkl', 'rb'))
    be = pickle.load(open('Results/drosophila_mentions_be_real.pkl', 'rb'))

    if not os.path.isfile('Results/train_test/dros_tr_te_split_144_train_data.pkl'):
        for seed in random_seeds:
            real_tr_te_name = 'train_test/dros_tr_te_split_' + str(seed)
            train_data, b, c, d = pred.manual_train_test_split(strict,
                                                               real_tr_te_name,
                                                               random_state=seed,
                                                               test_set_prop=0.1)
            tr_val_name = 'train_val/dros_tr_val_split_' + str(seed)
            real_train_data, b, c, d = pred.manual_train_test_split(train_data,
                                                                    tr_val_name,
                                                                    random_state=seed,
                                                                    test_set_prop=0.2)
    return strict, gen, be

def make_w2v_model(dataset, name_for_model, model_features=None):
    """Produce a Word2Vec Model

    Model_features (list): Features of the word to vec models
        1. Word vector dimensionality
        2. Minimum word count
        3. Number of threads to run in parallel
        4. Context window size
        5. Downsample setting for frequent words

    """

    print ('Parsing datasets sentences')

    sentences = [fr.sentence_to_wordlist(sen) for sen in dataset]

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

    # Set values for various parameters
    if model_features:
        num_features = model_features[0] #300  # Word vector dimensionality
        min_word_count = model_features[1] #5  # Minimum word count
        num_workers = model_features[2] #4  # Number of threads to run in parallel
        context = model_features[3] #6  # Context window size
        downsampling = model_features[4] #0.001  # Downsample setting for frequent words

    print('Training Word2Vec Model')

    model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count=min_word_count, \
            window=context, sample=downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=False)

    model_name = 'Results/' + name_for_model + '_model'

    model.save(model_name)

    w2v_model = model

    return w2v_model

def modelfit(alg, train_vecs, train_labels, test_vecs, test_labels, w2v_model_type, useTrainCV=True, cv_folds=5):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train_vecs,
                              label=train_labels)
        cvresult = xgb.cv(xgb_param,
                          xgtrain,
                          num_boost_round=alg.get_params()['n_estimators'],
                          nfold=cv_folds,
                          metrics='auc',
                          early_stopping_rounds=50)
        alg.set_params(n_estimators=cvresult.shape[0])

    #fit the algorithm on the data
    alg.fit(train_vecs, train_labels, eval_metric='auc')

    #Predict training set:
    test_predictions = alg.predict(test_vecs)
    test_predprob = alg.predict_proba(test_vecs)[:, 1]

    #Metrics
    accuracy = metrics.accuracy_score(test_labels, test_predictions)
    roc_auc = metrics.roc_auc_score(test_labels, test_predprob)
    class_report = metrics.classification_report(test_labels, test_predictions)

    #Print Model report:
    print(w2v_model_type, '\nModel Report')
    print(w2v_model_type, 'Accuracy: %.4g' % accuracy)
    print(w2v_model_type, 'AUC Score (Train): %f' % roc_auc)
    print(w2v_model_type, 'Report \n', class_report)


#     feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
#     feat_imp.plot(kind='bar', title='Feature Importance')
#     plt.ylabel('Feature Importance Score')

    return accuracy, roc_auc, test_predictions, test_predprob, class_report

def dims(parameters):
    name_of_result = 'dims_parameter'
    xgb_clf = XGBClassifier(seed=24)
    auc_values_full = []
    for i in parameters:
        w2v_parameters = [i, 5, multiprocessing.cpu_count(), 5, 0.001]
        strict, gen, be = open_files()
        w2v_strict = make_w2v_model(strict, 'models/dims_strict_'+str(i)+'_',
                                         w2v_parameters)
        w2v_gen = make_w2v_model(gen, 'models/dims_gen_'+str(i)+'_',
                                      w2v_parameters)
        w2v_be = make_w2v_model(be, 'models/dims_be_'+str(i)+'_',
                                     w2v_parameters)
        auc_values = []
        for seed in random_seeds:
            data_name = 'Results/train_val/dros_tr_val_split_'+str(seed)
            train_data = pickle.load(open(data_name + '_train_data.pkl', 'rb'))
            train_labels = pickle.load(open(data_name + '_train_labels.pkl', 'rb'))
            validation_data = pickle.load(open(data_name + '_test_data.pkl', 'rb'))
            validation_labels = pickle.load(open(data_name + '_test_labels.pkl', 'rb'))

            w2v_train_vecs, w2v_val_vecs = pred.word_2_vec_feat_vecs(train_data,
                                                                     validation_data,
                                                                     w2v_strict,
                                                                     feature_count=i)

            strict_list_SR = [w2v_train_vecs, w2v_val_vecs,
                              train_labels, validation_labels]

            w2v_train_vecs, w2v_val_vecs = pred.word_2_vec_feat_vecs(train_data,
                                                                     validation_data,
                                                                     w2v_gen,
                                                                     feature_count=i)

            strict_list_GEN = [w2v_train_vecs, w2v_val_vecs,
                               train_labels, validation_labels]

            w2v_train_vecs, w2v_val_vecs = pred.word_2_vec_feat_vecs(train_data,
                                                                     validation_data,
                                                                     w2v_be,
                                                                     feature_count=i)

            strict_list_BE = [w2v_train_vecs, w2v_val_vecs,
                              train_labels, validation_labels]

            pickle.dump(strict_list_SR, open('Results/result_list/dros_strict_list_SR_dims_param_'+str(i)+'_'+str(seed)+'_results_list.pkl', 'wb'))
            pickle.dump(strict_list_GEN, open('Results/result_list/dros_strict_list_GEN_dims_param_'+str(i)+'_'+str(seed)+'_results_list.pkl', 'wb'))
            pickle.dump(strict_list_BE, open('Results/result_list/dros_strict_list_BE_dims_param_'+str(i)+'_'+str(seed)+'_results_list.pkl', 'wb'))

            strict_final_list = [strict_list_SR,
                                 strict_list_GEN,
                                 strict_list_BE]

            print ('\nPredicting\n')
            accuracy = []
            probs = []
            fpr = []
            tpr = []
            labels = []
            auc_score = []
            report = []

            for entry, model_name in zip(strict_final_list, ['SR '+str(seed), 'GEN '+str(seed), 'BE '+str(seed)]):
                accuracy_norm, auc_score_norm, pred_labels_norm, probs_norm, class_report_norm  = modelfit(xgb_clf,
                                                                                                           entry[0],
                                                                                                           entry[2],
                                                                                                           entry[1],
                                                                                                           entry[3],
                                                                                                           model_name)
                fpr_norm, tpr_norm, _ = roc_curve(entry[3], probs_norm)

                accuracy.append([accuracy_norm])
                probs.append([probs_norm])
                fpr.append([fpr_norm])
                tpr.append([tpr_norm])
                labels.append([pred_labels_norm])
                auc_score.append([auc_score_norm])
                report.append([class_report_norm])

            auc_values.append(auc_score)

            pickle.dump(accuracy, open('Results/metrics/'+name_of_result+'_accuracy_pickle_'+str(seed)+'.pkl',
                                     'wb'))
            pickle.dump(probs, open('Results/metrics/'+name_of_result+'_probs_pickle_'+str(seed)+'.pkl',
                                    'wb'))
            pickle.dump(fpr, open('Results/metrics/'+name_of_result+'_fpr_pickle_'+str(seed)+'.pkl',
                                  'wb'))
            pickle.dump(tpr, open('Results/metrics/'+name_of_result+'_tpr_pickle_'+str(seed)+'.pkl',
                                  'wb'))
            pickle.dump(labels, open('Results/metrics/'+name_of_result+'_labels_pickle_'+str(seed)+'.pkl',
                                     'wb'))
            pickle.dump(auc_score, open('Results/metrics/'+name_of_result+'_auc_score_pickle_'+str(seed)+'.pkl',
                                     'wb'))
            pickle.dump(report, open('Results/metrics/'+name_of_result+'_report_pickle_'+str(seed)+'.pkl',
                                     'wb'))

        auc_values_full.append(auc_values)
    auc_value_dict = {}
    for i in range(len(auc_values_full)):
        for j in range(len(auc_values_full[i])):
            auc_value_dict.setdefault('SR_'+str(i+1), []).append(auc_values_full[i][j][0])
            auc_value_dict.setdefault('GEN_'+str(i+1), []).append(auc_values_full[i][j][1])
            auc_value_dict.setdefault('BE_'+str(i+1), []).append(auc_values_full[i][j][2])

    max_aucs = {}

    for i in range(len(parameters)):
        single_max = np.max([np.max(auc_value_dict['SR_'+str(i+1)]),
                             np.max(auc_value_dict['GEN_'+str(i+1)]),
                             np.max(auc_value_dict['BE_'+str(i+1)])])
    best_parameter = max(max_aucs.items(), key=operator.itemgetter(1))[0]*100

    return best_parameter

def word_count(parameters, best_dims):
    name_of_result = 'word_count_parameter'
    xgb_clf = XGBClassifier(seed=24)
    auc_values_full = []
    for i in parameters:
        w2v_parameters = [best_dims, i, multiprocessing.cpu_count(), 5, 0.001]
        strict, gen, be = open_files()
        w2v_strict = make_w2v_model(strict, 'models/word_count_strict_'+str(i)+'_',
                                         w2v_parameters)
        w2v_gen = make_w2v_model(gen, 'models/word_count_gen_'+str(i)+'_',
                                      w2v_parameters)
        w2v_be = make_w2v_model(be, 'models/word_count_be_'+str(i)+'_',
                                     w2v_parameters)
        auc_values = []
        for seed in random_seeds:
            data_name = 'Results/train_val/dros_tr_val_split_'+str(seed)
            train_data = pickle.load(open(data_name + '_train_data.pkl', 'rb'))
            train_labels = pickle.load(open(data_name + '_train_labels.pkl', 'rb'))
            validation_data = pickle.load(open(data_name + '_test_data.pkl', 'rb'))
            validation_labels = pickle.load(open(data_name + '_test_labels.pkl', 'rb'))

            w2v_train_vecs, w2v_val_vecs = pred.word_2_vec_feat_vecs(train_data,
                                                                     validation_data,
                                                                     w2v_strict,
                                                                     feature_count=best_dims)

            strict_list_SR = [w2v_train_vecs, w2v_val_vecs,
                              train_labels, validation_labels]

            w2v_train_vecs, w2v_val_vecs = pred.word_2_vec_feat_vecs(train_data,
                                                                     validation_data,
                                                                     w2v_gen,
                                                                     feature_count=best_dims)

            strict_list_GEN = [w2v_train_vecs, w2v_val_vecs,
                               train_labels, validation_labels]

            w2v_train_vecs, w2v_val_vecs = pred.word_2_vec_feat_vecs(train_data,
                                                                     validation_data,
                                                                     w2v_be,
                                                                     feature_count=best_dims)
            strict_list_BE = [w2v_train_vecs, w2v_val_vecs,
                              train_labels, validation_labels]

            pickle.dump(strict_list_SR, open('Results/result_list/dros_strict_list_SR_word_count_param_'+str(i)+'_'+str(seed)+'_results_list.pkl', 'wb'))
            pickle.dump(strict_list_GEN, open('Results/result_list/dros_strict_list_GEN_word_count_param_'+str(i)+'_'+str(seed)+'_results_list.pkl', 'wb'))
            pickle.dump(strict_list_BE, open('Results/result_list/dros_strict_list_BE_word_count_param_'+str(i)+'_'+str(seed)+'_results_list.pkl', 'wb'))

            strict_final_list = [strict_list_SR,
                                 strict_list_GEN,
                                 strict_list_BE]

            print ('\nPredicting\n')
            accuracy = []
            probs = []
            fpr = []
            tpr = []
            labels = []
            auc_score = []
            report = []

            for entry, model_name in zip(strict_final_list, ['SR '+str(seed), 'GEN '+str(seed), 'BE '+str(seed)]):
                accuracy_norm, auc_score_norm, pred_labels_norm, probs_norm, class_report_norm  = modelfit(xgb_clf,
                                                                                                           entry[0],
                                                                                                           entry[2],
                                                                                                           entry[1],
                                                                                                           entry[3],
                                                                                                           model_name)
                fpr_norm, tpr_norm, _ = roc_curve(entry[3], probs_norm)

                accuracy.append([accuracy_norm])
                probs.append([probs_norm])
                fpr.append([fpr_norm])
                tpr.append([tpr_norm])
                labels.append([pred_labels_norm])
                auc_score.append([auc_score_norm])
                report.append([class_report_norm])

            auc_values.append(auc_score)

            pickle.dump(accuracy, open('Results/metrics/'+name_of_result+'_accuracy_pickle_'+str(seed)+'.pkl',
                                     'wb'))
            pickle.dump(probs, open('Results/metrics/'+name_of_result+'_probs_pickle_'+str(seed)+'.pkl',
                                    'wb'))
            pickle.dump(fpr, open('Results/metrics/'+name_of_result+'_fpr_pickle_'+str(seed)+'.pkl',
                                  'wb'))
            pickle.dump(tpr, open('Results/metrics/'+name_of_result+'_tpr_pickle_'+str(seed)+'.pkl',
                                  'wb'))
            pickle.dump(labels, open('Results/metrics/'+name_of_result+'_labels_pickle_'+str(seed)+'.pkl',
                                     'wb'))
            pickle.dump(auc_score, open('Results/metrics/'+name_of_result+'_auc_score_pickle_'+str(seed)+'.pkl',
                                     'wb'))
            pickle.dump(report, open('Results/metrics/'+name_of_result+'_report_pickle_'+str(seed)+'.pkl',
                                     'wb'))

        auc_values_full.append(auc_values)
    auc_value_dict = {}
    for i in range(len(auc_values_full)):
        for j in range(len(auc_values_full[i])):
            auc_value_dict.setdefault('SR_'+str(i+1), []).append(auc_values_full[i][j][0])
            auc_value_dict.setdefault('GEN_'+str(i+1), []).append(auc_values_full[i][j][1])
            auc_value_dict.setdefault('BE_'+str(i+1), []).append(auc_values_full[i][j][2])

    max_aucs = {}

    for i in range(len(parameters)):
        single_max = np.max([np.max(auc_value_dict['SR_'+str(i+1)]),
                             np.max(auc_value_dict['GEN_'+str(i+1)]),
                             np.max(auc_value_dict['BE_'+str(i+1)])])
    best_parameter = max(max_aucs.items(), key=operator.itemgetter(1))[0]

    return best_parameter


def context_window(parameters, best_dims, best_word_count):
    name_of_result = 'context_window_parameter'
    xgb_clf = XGBClassifier(seed=24)
    auc_values_full = []
    for i in parameters:
        w2v_parameters = [best_dims, best_word_count, multiprocessing.cpu_count(), i, 0.001]
        strict, gen, be = open_files()
        w2v_strict = make_w2v_model(strict, 'models/context_window_strict_'+str(i)+'_',
                                         w2v_parameters)
        w2v_gen = make_w2v_model(gen, 'models/context_window_gen_'+str(i)+'_',
                                      w2v_parameters)
        w2v_be = make_w2v_model(be, 'models/context_window_be_'+str(i)+'_',
                                     w2v_parameters)
        auc_values = []
        for seed in random_seeds:
            data_name = 'Results/train_val/dros_tr_val_split_'+str(seed)
            train_data = pickle.load(open(data_name + '_train_data.pkl', 'rb'))
            train_labels = pickle.load(open(data_name + '_train_labels.pkl', 'rb'))
            validation_data = pickle.load(open(data_name + '_test_data.pkl', 'rb'))
            validation_labels = pickle.load(open(data_name + '_test_labels.pkl', 'rb'))

            w2v_train_vecs, w2v_val_vecs = pred.word_2_vec_feat_vecs(train_data,
                                                                     validation_data,
                                                                     w2v_strict,
                                                                     feature_count=best_dims)

            strict_list_SR = [w2v_train_vecs, w2v_val_vecs,
                              train_labels, validation_labels]

            w2v_train_vecs, w2v_val_vecs = pred.word_2_vec_feat_vecs(train_data,
                                                                     validation_data,
                                                                     w2v_gen,
                                                                     feature_count=best_dims)

            strict_list_GEN = [w2v_train_vecs, w2v_val_vecs,
                               train_labels, validation_labels]

            w2v_train_vecs, w2v_val_vecs = pred.word_2_vec_feat_vecs(train_data,
                                                                     validation_data,
                                                                     w2v_be,
                                                                     feature_count=best_dims)
            strict_list_BE = [w2v_train_vecs, w2v_val_vecs,
                              train_labels, validation_labels]

            pickle.dump(strict_list_SR, open('Results/result_list/dros_strict_list_SR_context_window_param_'+str(i)+'_'+str(seed)+'_results_list.pkl', 'wb'))
            pickle.dump(strict_list_GEN, open('Results/result_list/dros_strict_list_GEN_context_window_param_'+str(i)+'_'+str(seed)+'_results_list.pkl', 'wb'))
            pickle.dump(strict_list_BE, open('Results/result_list/dros_strict_list_BE_context_window_param_'+str(i)+'_'+str(seed)+'_results_list.pkl', 'wb'))

            strict_final_list = [strict_list_SR,
                                 strict_list_GEN,
                                 strict_list_BE]

            print ('\nPredicting\n')
            accuracy = []
            probs = []
            fpr = []
            tpr = []
            labels = []
            auc_score = []
            report = []

            for entry, model_name in zip(strict_final_list, ['SR '+str(seed), 'GEN '+str(seed), 'BE '+str(seed)]):
                accuracy_norm, auc_score_norm, pred_labels_norm, probs_norm, class_report_norm  = modelfit(xgb_clf,
                                                                                                           entry[0],
                                                                                                           entry[2],
                                                                                                           entry[1],
                                                                                                           entry[3],
                                                                                                           model_name)
                fpr_norm, tpr_norm, _ = roc_curve(entry[3], probs_norm)

                accuracy.append([accuracy_norm])
                probs.append([probs_norm])
                fpr.append([fpr_norm])
                tpr.append([tpr_norm])
                labels.append([pred_labels_norm])
                auc_score.append([auc_score_norm])
                report.append([class_report_norm])

            auc_values.append(auc_score)

            pickle.dump(accuracy, open('Results/metrics/'+name_of_result+'_accuracy_pickle_'+str(seed)+'.pkl',
                                     'wb'))
            pickle.dump(probs, open('Results/metrics/'+name_of_result+'_probs_pickle_'+str(seed)+'.pkl',
                                    'wb'))
            pickle.dump(fpr, open('Results/metrics/'+name_of_result+'_fpr_pickle_'+str(seed)+'.pkl',
                                  'wb'))
            pickle.dump(tpr, open('Results/metrics/'+name_of_result+'_tpr_pickle_'+str(seed)+'.pkl',
                                  'wb'))
            pickle.dump(labels, open('Results/metrics/'+name_of_result+'_labels_pickle_'+str(seed)+'.pkl',
                                     'wb'))
            pickle.dump(auc_score, open('Results/metrics/'+name_of_result+'_auc_score_pickle_'+str(seed)+'.pkl',
                                     'wb'))
            pickle.dump(report, open('Results/metrics/'+name_of_result+'_report_pickle_'+str(seed)+'.pkl',
                                     'wb'))

        auc_values_full.append(auc_values)
    auc_value_dict = {}
    for i in range(len(auc_values_full)):
        for j in range(len(auc_values_full[i])):
            auc_value_dict.setdefault('SR_'+str(i+1), []).append(auc_values_full[i][j][0])
            auc_value_dict.setdefault('GEN_'+str(i+1), []).append(auc_values_full[i][j][1])
            auc_value_dict.setdefault('BE_'+str(i+1), []).append(auc_values_full[i][j][2])

    max_aucs = {}

    for i in range(len(parameters)):
        single_max = np.max([np.max(auc_value_dict['SR_'+str(i+1)]),
                             np.max(auc_value_dict['GEN_'+str(i+1)]),
                             np.max(auc_value_dict['BE_'+str(i+1)])])
    best_parameter = max(max_aucs.items(), key=operator.itemgetter(1))[0]

    return best_parameter

def downsmpl(parameters, best_dims, best_word_count, best_context_window):
    name_of_result = 'context_window_parameter'
    xgb_clf = XGBClassifier(seed=24)
    auc_values_full = []
    for i in parameters:
        w2v_parameters = [best_dims,
                          best_word_count,
                          multiprocessing.cpu_count(),
                          best_context_window, i[1]]
        strict, gen, be = open_files()
        w2v_strict = make_w2v_model(strict, 'models/context_window_strict_'+str(i[0])+'_',
                                         w2v_parameters)
        w2v_gen = make_w2v_model(gen, 'models/context_window_gen_'+str(i[0])+'_',
                                      w2v_parameters)
        w2v_be = make_w2v_model(be, 'models/context_window_be_'+str(i[0])+'_',
                                     w2v_parameters)
        auc_values = []
        for seed in random_seeds:
            data_name = 'Results/train_val/dros_tr_val_split_'+str(seed)
            train_data = pickle.load(open(data_name + '_train_data.pkl', 'rb'))
            train_labels = pickle.load(open(data_name + '_train_labels.pkl', 'rb'))
            validation_data = pickle.load(open(data_name + '_test_data.pkl', 'rb'))
            validation_labels = pickle.load(open(data_name + '_test_labels.pkl', 'rb'))

            w2v_train_vecs, w2v_val_vecs = pred.word_2_vec_feat_vecs(train_data,
                                                                     validation_data,
                                                                     w2v_strict,
                                                                     feature_count=best_dims)

            strict_list_SR = [w2v_train_vecs, w2v_val_vecs,
                              train_labels, validation_labels]

            w2v_train_vecs, w2v_val_vecs = pred.word_2_vec_feat_vecs(train_data,
                                                                     validation_data,
                                                                     w2v_gen,
                                                                     feature_count=best_dims)

            strict_list_GEN = [w2v_train_vecs, w2v_val_vecs,
                               train_labels, validation_labels]

            w2v_train_vecs, w2v_val_vecs = pred.word_2_vec_feat_vecs(train_data,
                                                                     validation_data,
                                                                     w2v_be,
                                                                     feature_count=best_dims)
            strict_list_BE = [w2v_train_vecs, w2v_val_vecs,
                              train_labels, validation_labels]

            pickle.dump(strict_list_SR, open('Results/result_list/dros_strict_list_SR_context_window_param_'+str(i[0])+'_'+str(seed)+'_results_list.pkl', 'wb'))
            pickle.dump(strict_list_GEN, open('Results/result_list/dros_strict_list_GEN_context_window_param_'+str(i[0])+'_'+str(seed)+'_results_list.pkl', 'wb'))
            pickle.dump(strict_list_BE, open('Results/result_list/dros_strict_list_BE_context_window_param_'+str(i[0])+'_'+str(seed)+'_results_list.pkl', 'wb'))

            strict_final_list = [strict_list_SR,
                                 strict_list_GEN,
                                 strict_list_BE]

            print ('\nPredicting\n')
            accuracy = []
            probs = []
            fpr = []
            tpr = []
            labels = []
            auc_score = []
            report = []

            for entry, model_name in zip(strict_final_list, ['SR '+str(seed), 'GEN '+str(seed), 'BE '+str(seed)]):
                accuracy_norm, auc_score_norm, pred_labels_norm, probs_norm, class_report_norm  = modelfit(xgb_clf,
                                                                                                           entry[0],
                                                                                                           entry[2],
                                                                                                           entry[1],
                                                                                                           entry[3],
                                                                                                           model_name)
                fpr_norm, tpr_norm, _ = roc_curve(entry[3], probs_norm)

                accuracy.append([accuracy_norm])
                probs.append([probs_norm])
                fpr.append([fpr_norm])
                tpr.append([tpr_norm])
                labels.append([pred_labels_norm])
                auc_score.append([auc_score_norm])
                report.append([class_report_norm])

            auc_values.append(auc_score)

            pickle.dump(accuracy, open('Results/metrics/'+name_of_result+'_accuracy_pickle_'+str(seed)+'.pkl',
                                     'wb'))
            pickle.dump(probs, open('Results/metrics/'+name_of_result+'_probs_pickle_'+str(seed)+'.pkl',
                                    'wb'))
            pickle.dump(fpr, open('Results/metrics/'+name_of_result+'_fpr_pickle_'+str(seed)+'.pkl',
                                  'wb'))
            pickle.dump(tpr, open('Results/metrics/'+name_of_result+'_tpr_pickle_'+str(seed)+'.pkl',
                                  'wb'))
            pickle.dump(labels, open('Results/metrics/'+name_of_result+'_labels_pickle_'+str(seed)+'.pkl',
                                     'wb'))
            pickle.dump(auc_score, open('Results/metrics/'+name_of_result+'_auc_score_pickle_'+str(seed)+'.pkl',
                                     'wb'))
            pickle.dump(report, open('Results/metrics/'+name_of_result+'_report_pickle_'+str(seed)+'.pkl',
                                     'wb'))

        auc_values_full.append(auc_values)
    auc_value_dict = {}
    for i in range(len(auc_values_full)):
        for j in range(len(auc_values_full[i])):
            auc_value_dict.setdefault('SR_'+str(i+1), []).append(auc_values_full[i][j][0])
            auc_value_dict.setdefault('GEN_'+str(i+1), []).append(auc_values_full[i][j][1])
            auc_value_dict.setdefault('BE_'+str(i+1), []).append(auc_values_full[i][j][2])

    max_aucs = {}

    for i in range(len(parameters)):
        single_max = np.max([np.max(auc_value_dict['SR_'+str(i+1)]),
                             np.max(auc_value_dict['GEN_'+str(i+1)]),
                             np.max(auc_value_dict['BE_'+str(i+1)])])
    best_parameter = max(max_aucs.items(), key=operator.itemgetter(1))[0]

    return best_parameter







def main():
    dimensions = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    min_word_count = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    context_win = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    downsample = [(1, 0.1), (2, 0.01), (3, 0.001), (4, 0.0001), (5, 0.00001), (6, 0.000001)]

    best_dims = dims(dimensions)
    best_word_count = word_count(min_word_count, best_dims)
    best_context_window = context_window(context_win, best_dims, best_word_count)
    best_donwsample = downsmpl(downsample, best_dims, best_word_count, best_context_window)

    best_parameters = [best_dims, best_word_count, best_context_window, downsample[best_donwsample-1]]

    pickle.dump(best_parameters, open('Results/Best_Word2Vec_paramerts.pkl', 'wb'))

if __name__ == '__main__':
    main()
