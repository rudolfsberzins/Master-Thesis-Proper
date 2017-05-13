import pickle
import os
import re
import numpy as np
from gensim.models import word2vec
import logging
import pandas as pd
from parse_and_prepare import ProteinProteinInteractionClassifier as ppi
import file_readers as fr
import prediction as pred
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

def modelfit(alg, train_vecs, train_labels, w2v_model_type, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param=alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train_vecs,
                              label=train_labels)
        cvresult = xgb.cv(xgb_param,
                          xgtrain,
                          num_boost_round=alg.get_params()['n_estimators'],
                          nfold=cv_folds,
                          metrics='auc',
                          early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    #fit the algorithm on the data
    alg.fit(train_vecs, train_labels, eval_metric='auc')

    #Predict training set:
    train_predictions = alg.predict(train_vecs)
    train_predprob = alg.predict_proba(train_vecs)[:,1]

    #Print Model report:
    print(w2v_model_type, '\nModel Report')
    print(w2v_model_type, 'Accuracy: %.4g' % metrics.accuracy_score(train_labels, train_predictions))
    print(w2v_model_type, 'AUC Score (Train): %f' % metrics.roc_auc_score(train_labels, train_predprob))

    error = 1-metrics.accuracy_score(train_labels, train_predictions)
    auc = metrics.roc_auc_score(train_labels, train_predprob)

    return error, auc

def most_common(lst):
    return max(set(lst), key=lst.count)

def main():
    print('\nPreparing Files\n')
    yeast_strict_real = pickle.load(open('Results/yeast_mentions_strict_real.pkl', 'rb'))
    yeast_gen_real = pickle.load(open('Results/yeast_mentions_gen_real.pkl', 'rb'))
    yeast_be_real = pickle.load(open('Results/yeast_mentions_be_real.pkl', 'rb'))
    random_seeds = [144, 235, 905, 2895, 3462, 4225, 5056, 5192, 7751, 7813]

    yeast_w2v_model_strict = pred.make_w2v_model(yeast_strict_real, 'yeast_strict')
    yeast_w2v_model_gen = pred.make_w2v_model(yeast_gen_real, 'yeast_gen')
    yeast_w2v_model_be = pred.make_w2v_model(yeast_be_real, 'yeast_be')

    for seed in random_seeds:
        real_tr_te_name = 'yeast_tr_te_split_' + str(seed)
        train_data, test_data, train_labels, test_labels = pred.manual_train_test_split(yeast_strict_real, real_tr_te_name, random_state=seed ,test_set_prop=0.1)

        w2v_train_vecs, w2v_test_vecs = pred.word_2_vec_feat_vecs(train_data, test_data, yeast_w2v_model_strict, feature_count=600)

        strict_list_SR_dims_param = [w2v_train_vecs, w2v_test_vecs,
                                     train_labels, test_labels]

        w2v_train_vecs, w2v_test_vecs = pred.word_2_vec_feat_vecs(train_data, test_data, yeast_w2v_model_gen, feature_count=600)

        strict_list_GEN_dims_param = [w2v_train_vecs, w2v_test_vecs,
                                      train_labels, test_labels]

        w2v_train_vecs, w2v_test_vecs = pred.word_2_vec_feat_vecs(train_data, test_data, yeast_w2v_model_be, feature_count=600)

        strict_list_BE_dims_param = [w2v_train_vecs, w2v_test_vecs,
                                     train_labels, test_labels]

        pickle.dump(strict_list_SR_dims_param, open('Results/yeast_strict_list_SR_dims_param_'+str(seed)+'_results_list.pkl', 'wb'))
        pickle.dump(strict_list_GEN_dims_param, open('Results/yeast_strict_list_GEN_dims_param_'+str(seed)+'_results_list.pkl', 'wb'))
        pickle.dump(strict_list_BE_dims_param, open('Results/yeast_strict_list_BE_dims_param_'+str(seed)+'_results_list.pkl', 'wb'))

    print('\nFirst XGB run\n')
    xgb1 = XGBClassifier(learning_rate=0.1,
                         n_estimators=1000,
                         max_depth=5,
                         min_child_weight=1,
                         gamma=0,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         objective='binary:logistic',
                         nthread=4,
                         scale_pos_weight=1,
                         seed=24)

    strict_error_list = []
    strict_auc_list = []
    gen_error_list = []
    gen_auc_list = []
    be_error_list = []
    be_auc_list = []

    for seed in random_seeds:

        strict_list = pickle.load(open('Results/yeast_strict_list_SR_dims_param_'+str(seed)+'_results_list.pkl', 'rb'))
        gen_list = pickle.load(open('Results/yeast_strict_list_GEN_dims_param_'+str(seed)+'_results_list.pkl', 'rb'))
        be_list = pickle.load(open('Results/yeast_strict_list_BE_dims_param_'+str(seed)+'_results_list.pkl', 'rb'))
        print('\n', seed, '\n')
        strict_error, strict_auc = modelfit(xgb1, strict_list[0], strict_list[2], 'STRICT')
        gen_error, gen_auc = modelfit(xgb1, gen_list[0], gen_list[2], 'GEN')
        be_error, be_auc = modelfit(xgb1, be_list[0], be_list[2], 'BE')

        strict_error_list.append(strict_error)
        strict_auc_list.append(strict_auc)
        gen_error_list.append(gen_error)
        gen_auc_list.append(gen_auc)
        be_error_list.append(be_error)
        be_auc_list.append(be_auc)

    print('\n')
    print('Strict Results Error: ', np.mean(strict_error_list), ', Auc: ', np.mean(strict_auc_list))
    print('Gen Results Error: ', np.mean(gen_error_list), ', Auc: ', np.mean(gen_auc_list))
    print('Be Results Error: ', np.mean(be_error_list), ', Auc: ', np.mean(be_auc_list))
    print('\n')

    print('\nFirst param_test\n')
    param_test1 = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'min_child_weight': [1, 2, 3, 4, 5, 6]
    }
    gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1,
                                                      n_estimators=140,
                                                      max_depth=5,
                                                      min_child_weight=1,
                                                      gamma=0,
                                                      subsample=0.8,
                                                      colsample_bytree=0.8,
                                                      objective='binary:logistic',
                                                      nthread=4,
                                                      scale_pos_weight=1,
                                                      seed=24),
                            param_grid=param_test1,
                            scoring='roc_auc',
                            n_jobs=4,
                            iid=False,
                            cv=5)

    strict_param_test1 = []
    for seed in random_seeds:
        strict_list = pickle.load(open('Results/yeast_strict_list_SR_dims_param_'+str(seed)+'_results_list.pkl', 'rb'))
        gsearch1.fit(strict_list[0], strict_list[2])
        strict_param_test1.append((seed, (gsearch1.best_params_, gsearch1.best_score_)))

    gen_param_test1 = []
    for seed in random_seeds:
        gen_list = pickle.load(open('Results/yeast_strict_list_GEN_dims_param_'+str(seed)+'_results_list.pkl', 'rb'))
        gsearch1.fit(gen_list[0], gen_list[2])
        gen_param_test1.append((seed, (gsearch1.best_params_, gsearch1.best_score_)))

    be_param_test1 = []
    for seed in random_seeds:
        be_list = pickle.load(open('Results/yeast_strict_list_BE_dims_param_'+str(seed)+'_results_list.pkl', 'rb'))
        gsearch1.fit(be_list[0], be_list[2])
        be_param_test1.append((seed, (gsearch1.best_params_, gsearch1.best_score_)))


    strict_max_depth = []
    strict_min_child_weigth = []
    gen_max_depth = []
    gen_min_child_weight = []
    be_max_depth = []
    be_min_child_weight = []
    for i in range(10):
        print('\n', strict_param_test1[i], '\n', gen_param_test1[i], '\n', be_param_test1[i], '\n')

        strict_max_depth.append(strict_param_test1[i][1][0]['max_depth'])
        strict_min_child_weigth.append(strict_param_test1[i][1][0]['min_child_weight'])
        gen_max_depth.append(gen_param_test1[i][1][0]['max_depth'])
        gen_min_child_weight.append(gen_param_test1[i][1][0]['min_child_weight'])
        be_max_depth.append(be_param_test1[i][1][0]['max_depth'])
        be_min_child_weight.append(be_param_test1[i][1][0]['min_child_weight'])

    best_overall_max_depth = []
    best_overall_min_child_weight = []
    best_overall_max_depth.append(most_common(strict_max_depth), most_common(gen_max_depth), most_common(be_max_depth))
    best_overall_min_child_weight.append(most_common(strict_min_child_weigth), most_common(gen_min_child_weight), most_common(be_min_child_weight))
    print('Best overall max_depth: Strict - ', best_overall_max_depth[0], ', Gen - ', best_overall_max_depth[1], ', Be - ', best_overall_max_depth[2])
    print('Best overall min_child_weight: Strict - ', best_overall_min_child_weight[0], ', Gen - ', best_overall_min_child_weight[1], ', Be - ', best_overall_min_child_weight[2])
    best_max_depth = most_common(best_overall_max_depth)
    best_min_child_weight = most_common(best_overall_min_child_weight)

    print('Optimum max_depth:', best_max_depth, ' Optimum min_child_weight: ', best_min_child_weight)

    print('\nSecond param_test\n')
    param_test2 = {
        'gamma':[i/10.0 for i in range(0, 5)]
    }
    gsearch2 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1,
                                                      n_estimators=140,
                                                      max_depth=best_max_depth,
                                                      min_child_weight=best_min_child_weight,
                                                      gamma=0,
                                                      subsample=0.8,
                                                      colsample_bytree=0.8,
                                                      objective='binary:logistic',
                                                      nthread=4,
                                                      scale_pos_weight=1,
                                                      seed=24),
                            param_grid=param_test2,
                            scoring='roc_auc',
                            n_jobs=4,
                            iid=False,
                            cv=5)

    strict_param_test2 = []
    for seed in random_seeds:
        strict_list = pickle.load(open('Results/yeast_strict_list_SR_dims_param_'+str(seed)+'_results_list.pkl', 'rb'))
        gsearch2.fit(strict_list[0], strict_list[2])
        strict_param_test2.append((seed, (gsearch2.best_params_, gsearch2.best_score_)))

    gen_param_test2 = []
    for seed in random_seeds:
        gen_list = pickle.load(open('Results/yeast_strict_list_GEN_dims_param_'+str(seed)+'_results_list.pkl', 'rb'))
        gsearch2.fit(gen_list[0], gen_list[2])
        gen_param_test2.append((seed, (gsearch2.best_params_, gsearch2.best_score_)))

    be_param_test2 = []
    for seed in random_seeds:
        be_list = pickle.load(open('Results/yeast_strict_list_BE_dims_param_'+str(seed)+'_results_list.pkl', 'rb'))
        gsearch2.fit(be_list[0], be_list[2])
        be_param_test2.append((seed, (gsearch2.best_params_, gsearch2.best_score_)))


    strict_gamma = []
    gen_gamma = []
    be_gamma = []
    for i in range(10):
        print('\n', strict_param_test2[i], '\n', gen_param_test2[i], '\n', be_param_test2[i], '\n')

        strict_gamma.append(strict_param_test2[i][1][0]['gamma'])
        gen_gamma.append(gen_param_test2[i][1][0]['gamma'])
        be_gamma.append(be_param_test2[i][1][0]['gamma'])

    best_overall_gamma = []
    best_overall_gamma.append(most_common(strict_gamma), most_common(gen_gamma), most_common(be_gamma))
    print('Best overall gamma: Strict - ', best_overall_gamma[0], ', Gen - ', best_overall_gamma[1], ', Be - ', best_overall_gamma[2])
    best_gamma = most_common(best_overall_gamma)

    print('Optimum gamma:', best_gamma)
    print('\nSecond XGB run\n')
    xgb2 = XGBClassifier(learning_rate=0.1,
                         n_estimators=1000,
                         max_depth=best_max_depth,
                         min_child_weight=best_min_child_weight,
                         gamma=best_gamma,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         objective='binary:logistic',
                         nthread=4,
                         scale_pos_weight=1,
                         seed=24)

    strict_error_list = []
    strict_auc_list = []
    gen_error_list = []
    gen_auc_list = []
    be_error_list = []
    be_auc_list = []

    for seed in random_seeds:

        strict_list = pickle.load(open('Results/yeast_strict_list_SR_dims_param_'+str(seed)+'_results_list.pkl', 'rb'))
        gen_list = pickle.load(open('Results/yeast_strict_list_GEN_dims_param_'+str(seed)+'_results_list.pkl', 'rb'))
        be_list = pickle.load(open('Results/yeast_strict_list_BE_dims_param_'+str(seed)+'_results_list.pkl', 'rb'))
        print('\n', seed, '\n')
        strict_error, strict_auc = modelfit(xgb2, strict_list[0], strict_list[2], 'STRICT')
        gen_error, gen_auc = modelfit(xgb2, gen_list[0], gen_list[2], 'GEN')
        be_error, be_auc = modelfit(xgb2, be_list[0], be_list[2], 'BE')

        strict_error_list.append(strict_error)
        strict_auc_list.append(strict_auc)
        gen_error_list.append(gen_error)
        gen_auc_list.append(gen_auc)
        be_error_list.append(be_error)
        be_auc_list.append(be_auc)

    print('\n')
    print('Strict Results Error: ', np.mean(strict_error_list), ', Auc: ', np.mean(strict_auc_list))
    print('Gen Results Error: ', np.mean(gen_error_list), ', Auc: ', np.mean(gen_auc_list))
    print('Be Results Error: ', np.mean(be_error_list), ', Auc: ', np.mean(be_auc_list))
    print('\n')

    print('\nThird param_test\n')
    param_test3 = {
        'subsample': [i/10.0 for i in range(6,10)],
        'colsample_bytree': [i/10.0 for i in range(6,10)]
    }
    gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1,
                                                      n_estimators=140,
                                                      max_depth=best_max_depth,
                                                      min_child_weight=best_min_child_weight,
                                                      gamma=best_gamma,
                                                      subsample=0.8,
                                                      colsample_bytree=0.8,
                                                      objective='binary:logistic',
                                                      nthread=4,
                                                      scale_pos_weight=1,
                                                      seed=24),
                            param_grid=param_test3,
                            scoring='roc_auc',
                            n_jobs=4,
                            iid=False,
                            cv=5)

    strict_param_test3 = []
    for seed in random_seeds:
        strict_list = pickle.load(open('Results/yeast_strict_list_SR_dims_param_'+str(seed)+'_results_list.pkl', 'rb'))
        gsearch3.fit(strict_list[0], strict_list[2])
        strict_param_test3.append((seed, (gsearch3.best_params_, gsearch3.best_score_)))

    gen_param_test3 = []
    for seed in random_seeds:
        gen_list = pickle.load(open('Results/yeast_strict_list_GEN_dims_param_'+str(seed)+'_results_list.pkl', 'rb'))
        gsearch3.fit(gen_list[0], gen_list[2])
        gen_param_test3.append((seed, (gsearch3.best_params_, gsearch3.best_score_)))

    be_param_test3 = []
    for seed in random_seeds:
        be_list = pickle.load(open('Results/yeast_strict_list_BE_dims_param_'+str(seed)+'_results_list.pkl', 'rb'))
        gsearch3.fit(be_list[0], be_list[2])
        be_param_test3.append((seed, (gsearch3.best_params_, gsearch3.best_score_)))


    strict_subsample = []
    strict_colsample_bytree = []
    gen_subsample = []
    gen_colsample_bytree = []
    be_subsample = []
    be_colsample_bytree = []
    for i in range(10):
        print('\n', strict_param_test3[i], '\n', gen_param_test3[i], '\n', be_param_test3[i], '\n')

        strict_subsample.append(strict_param_test3[i][1][0]['subsample'])
        strict_colsample_bytree.append(strict_param_test3[i][1][0]['colsample_bytree'])
        gen_subsample.append(gen_param_test3[i][1][0]['subsample'])
        gen_colsample_bytree.append(gen_param_test3[i][1][0]['colsample_bytree'])
        be_subsample.append(be_param_test3[i][1][0]['subsample'])
        be_colsample_bytree.append(be_param_test3[i][1][0]['colsample_bytree'])

    best_overall_subsample = []
    best_overall_colsample_bytree = []
    best_overall_subsample.append(most_common(strict_subsample), most_common(gen_subsample), most_common(be_subsample))
    best_overall_colsample_bytree.append(most_common(strict_min_child_weigth), most_common(gen_colsample_bytree), most_common(be_colsample_bytree))
    print('Best overall subsample: Strict - ', best_overall_subsample[0], ', Gen - ', best_overall_subsample[1], ', Be - ', best_overall_subsample[2])
    print('Best overall colsample_bytree: Strict - ', best_overall_colsample_bytree[0], ', Gen - ', best_overall_colsample_bytree[1], ', Be - ', best_overall_colsample_bytree[2])
    best_subsample = most_common(best_overall_subsample)
    best_colsample_bytree = most_common(best_overall_colsample_bytree)

    print('Optimum subsample:', best_subsample, ' Optimum colsample_bytree: ', best_colsample_bytree)

    print('\nFourth param_test\n')
    param_test4 = {
        'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05, 0.1, 1, 100]
    }
    gsearch4 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1,
                                                      n_estimators=140,
                                                      max_depth=best_max_depth,
                                                      min_child_weight=best_min_child_weight,
                                                      gamma=best_gamma,
                                                      subsample=best_subsample,
                                                      colsample_bytree=best_colsample_bytree,
                                                      objective='binary:logistic',
                                                      nthread=4,
                                                      scale_pos_weight=1,
                                                      seed=24),
                            param_grid=param_test4,
                            scoring='roc_auc',
                            n_jobs=4,
                            iid=False,
                            cv=5)

    strict_param_test4 = []
    for seed in random_seeds:
        strict_list = pickle.load(open('Results/yeast_strict_list_SR_dims_param_'+str(seed)+'_results_list.pkl', 'rb'))
        gsearch4.fit(strict_list[0], strict_list[2])
        strict_param_test4.append((seed, (gsearch4.best_params_, gsearch4.best_score_)))

    gen_param_test4 = []
    for seed in random_seeds:
        gen_list = pickle.load(open('Results/yeast_strict_list_GEN_dims_param_'+str(seed)+'_results_list.pkl', 'rb'))
        gsearch4.fit(gen_list[0], gen_list[2])
        gen_param_test4.append((seed, (gsearch4.best_params_, gsearch4.best_score_)))

    be_param_test4 = []
    for seed in random_seeds:
        be_list = pickle.load(open('Results/yeast_strict_list_BE_dims_param_'+str(seed)+'_results_list.pkl', 'rb'))
        gsearch4.fit(be_list[0], be_list[2])
        be_param_test4.append((seed, (gsearch4.best_params_, gsearch4.best_score_)))


    strict_reg_alpha = []
    gen_reg_alpha = []
    be_reg_alpha = []
    for i in range(10):
        print('\n', strict_param_test4[i], '\n', gen_param_test4[i], '\n', be_param_test4[i], '\n')

        strict_reg_alpha.append(strict_param_test4[i][1][0]['reg_alpha'])
        gen_reg_alpha.append(gen_param_test4[i][1][0]['reg_alpha'])
        be_reg_alpha.append(be_param_test4[i][1][0]['reg_alpha'])

    best_overall_reg_alpha = []
    best_overall_reg_alpha.append(most_common(strict_reg_alpha), most_common(gen_reg_alpha), most_common(be_reg_alpha))
    print('Best overall reg_alpha: Strict - ', best_overall_reg_alpha[0], ', Gen - ', best_overall_reg_alpha[1], ', Be - ', best_overall_reg_alpha[2])
    best_reg_alpha = most_common(best_overall_reg_alpha)

    print('Optimum reg_alpha:', best_reg_alpha)

if __name__ == '__main__':
  main()






