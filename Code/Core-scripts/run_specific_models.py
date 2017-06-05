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

random_seeds = [144, 235, 905, 2895, 3462, 4225, 5056, 5192, 7751, 7813]

def open_files(organism):
    strict = pickle.load(open('Results/'+organism+'_mentions_strict_real.pkl', 'rb'))
    gen = pickle.load(open('Results/'+organism+'_mentions_gen_real.pkl', 'rb'))
    be = pickle.load(open('Results/'+organism+'_mentions_be_real.pkl', 'rb'))

    if not os.path.isfile('Results/train_test/'+organism+'_tr_te_split_144_train_data.pkl'):
        for seed in random_seeds:
            real_tr_te_name = 'train_test/'+organism+'_tr_te_split_' + str(seed)
            train_data, b, c, d = pred.manual_train_test_split(strict,
                                                               real_tr_te_name,
                                                               random_state=seed,
                                                               test_set_prop=0.1)

    return strict, gen, be

def run(organism, strict_model, gen_model, be_model, model_name):
    name_of_result = organism
    xgb_clf = XGBClassifier(seed=24)
    auc_values_full = []
    strict, gen, be = open_files(organism)
    w2v_strict = word2vec.Word2Vec.load(strict_model)
    w2v_gen = word2vec.Word2Vec.load(gen_model)
    w2v_be = word2vec.Word2Vec.load(be_model)
    for seed in random_seeds:
        data_name = 'Results/train_test/'+organism+'_tr_te_split_'+str(seed)
        train_data = pickle.load(open(data_name + '_train_data.pkl', 'rb'))
        train_labels = pickle.load(open(data_name + '_train_labels.pkl', 'rb'))
        validation_data = pickle.load(open(data_name + '_test_data.pkl', 'rb'))
        validation_labels = pickle.load(open(data_name + '_test_labels.pkl', 'rb'))

        w2v_train_vecs, w2v_val_vecs = pred.word_2_vec_feat_vecs(train_data,
                                                                 validation_data,
                                                                 w2v_strict,
                                                                 feature_count=800)

        strict_list_SR = [w2v_train_vecs, w2v_val_vecs,
                          train_labels, validation_labels]

        w2v_train_vecs, w2v_val_vecs = pred.word_2_vec_feat_vecs(train_data,
                                                                 validation_data,
                                                                 w2v_gen,
                                                                 feature_count=800)

        strict_list_GEN = [w2v_train_vecs, w2v_val_vecs,
                           train_labels, validation_labels]

        w2v_train_vecs, w2v_val_vecs = pred.word_2_vec_feat_vecs(train_data,
                                                                 validation_data,
                                                                 w2v_be,
                                                                 feature_count=800)

        strict_list_BE = [w2v_train_vecs, w2v_val_vecs,
                          train_labels, validation_labels]

        pickle.dump(strict_list_SR, open('Results/'+model_name+'/result_list/'+organism+'_strict_list_SR_'+str(seed)+'_results_list.pkl', 'wb'))
        pickle.dump(strict_list_GEN, open('Results/'+model_name+'/result_list/'+organism+'_strict_list_GEN_'+str(seed)+'_results_list.pkl', 'wb'))
        pickle.dump(strict_list_BE, open('Results/'+model_name+'/result_list/'+organism+'_strict_list_BE_'+str(seed)+'_results_list.pkl', 'wb'))

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

        for entry, name_model in zip(strict_final_list, ['SR '+str(seed), 'GEN '+str(seed), 'BE '+str(seed)]):
            accuracy_norm, auc_score_norm, pred_labels_norm, probs_norm, class_report_norm  = pred.XGB_modelfit(xgb_clf,
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

        pickle.dump(accuracy, open('Results/'+model_name+'/metrics/'+name_of_result+'_accuracy_pickle_'+str(seed)+'.pkl',
                                 'wb'))
        pickle.dump(probs, open('Results/'+model_name+'/metrics/'+name_of_result+'_probs_pickle_'+str(seed)+'.pkl',
                                'wb'))
        pickle.dump(fpr, open('Results/'+model_name+'/metrics/'+name_of_result+'_fpr_pickle_'+str(seed)+'.pkl',
                              'wb'))
        pickle.dump(tpr, open('Results/'+model_name+'/metrics/'+name_of_result+'_tpr_pickle_'+str(seed)+'.pkl',
                              'wb'))
        pickle.dump(labels, open('Results/'+model_name+'/metrics/'+name_of_result+'_labels_pickle_'+str(seed)+'.pkl',
                                 'wb'))
        pickle.dump(auc_score, open('Results/'+model_name+'/metrics/'+name_of_result+'_auc_score_pickle_'+str(seed)+'.pkl',
                                 'wb'))
        pickle.dump(report, open('Results/'+model_name+'/metrics/'+name_of_result+'_report_pickle_'+str(seed)+'.pkl',
                                 'wb'))

def main():
    _, organism, strict_model, gen_model, be_model, model_name = argv

    if not os.path.isdir('Results/'+model_name+'/result_list/'):
        os.makedirs('Results/'+model_name+'/result_list/')
    if not os.path.isdir('Results/'+model_name+'/metrics/'):
        os.makedirs('Results/'+model_name+'/metrics/')

    run(organism, strict_model, gen_model, be_model, model_name)

if __name__ == '__main__':
    main()

