from parse_and_prepare import ProteinProteinInteractionClassifier as ppi
import file_readers as fr
import prediction as pred
import pickle
import multiprocessing
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
import time

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def run(seq_tuple, model_name):
    name_of_results = model_name
    xgb_clf = XGBClassifier(seed=24)
    big_list = []
    w2v_model = word2vec.Word2Vec.load('Results/models/final_word2vec_model')
    for name in seq_tuple[0]:
        data_name = 'Results/strict_data/{}_mentions_strict_real.pkl'.format(name)
        data = pickle.load(open(data_name, 'rb'))
        big_list.append(data)
    test_org_name = 'Results/strict_data/{}_mentions_strict_real.pkl'.format(seq_tuple[1])
    test_org = pickle.load(open(test_org_name, 'rb'))

    train_orgs = merge_dicts(big_list[0], big_list[1], big_list[2], big_list[3])

    train_labels, test_labels = fr.produce_labels(train_orgs, test_org)

    w2v_train_vecs, w2v_test_vecs = pred.word_2_vec_feat_vecs(train_orgs, test_org, w2v_model)

    result_list = [w2v_train_vecs, w2v_test_vecs,
                   train_labels, test_labels]

    pickle.dump(result_list, open('Results/result_list/' + name_of_results + '_results_list.pkl', 'wb'))
    strict_final_list = [result_list]
    print ('\nPredicting\n')
    accuracy = []
    probs = []
    fpr = []
    tpr = []
    labels = []
    auc_score = []
    report = []

    for entry, name_model in zip(strict_final_list, [name_of_results]):
        accuracy_norm, auc_score_norm, pred_labels_norm, probs_norm, class_report_norm = pred.XGB_modelfit(xgb_clf,
                                                                                                           entry[0],
                                                                                                           entry[2],
                                                                                                           entry[1],
                                                                                                           entry[3],
                                                                                                           name_model)
        fpr_norm, tpr_norm, _ = roc_curve(entry[3], probs_norm)

        accuracy.append([accuracy_norm])
        probs.append([probs_norm])
        fpr.append([fpr_norm])
        tpr.append([tpr_norm])
        labels.append([pred_labels_norm])
        auc_score.append([auc_score_norm])
        report.append([class_report_norm])

    pickle.dump(accuracy, open('Results/metrics/'+name_of_results+'_accuracy_pickle_.pkl',
                               'wb'))
    pickle.dump(probs, open('Results/metrics/'+name_of_results+'_probs_pickle_.pkl',
                            'wb'))
    pickle.dump(fpr, open('Results/metrics/'+name_of_results+'_fpr_pickle_.pkl',
                          'wb'))
    pickle.dump(tpr, open('Results/metrics/'+name_of_results+'_tpr_pickle_.pkl',
                          'wb'))
    pickle.dump(labels, open('Results/metrics/'+name_of_results+'_labels_pickle_.pkl',
                             'wb'))
    pickle.dump(auc_score, open('Results/metrics/'+name_of_results+'_auc_score_pickle_.pkl',
                                'wb'))
    pickle.dump(report, open('Results/metrics/'+name_of_results+'_report_pickle_.pkl',
                             'wb'))

def main():
    _, org1, org2, org3, org4, odd_out, name = argv

    seq_tuple = ([org1, org2, org3, org4], odd_out)

    run(seq_tuple, name)

if __name__ == '__main__':
    main()
