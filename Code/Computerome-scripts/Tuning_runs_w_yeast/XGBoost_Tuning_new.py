# import sys
# sys.path.insert(0, '../../Core-scripts/')

import pickle
import os
import re
import numpy as np
from gensim.models import word2vec
import logging
import pandas as pd
import file_readers as fr
import prediction as pred
import xgboost as xgb
from sys import argv
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

def main():

    if not os.path.isdir('Results/GSCV/'):
        os.makedirs('Results/GSCV/')

    result_list_name = 'Results/result_list/yeast_out_results_list.pkl'
    result_list = pickle.load(open(result_list_name, 'rb'))

    parameters = {
        'learning_rate': [0.001, 0.01, 0.1, 0.3],
        'n_estimators': [100, 200, 500, 1000, 10000],
        'max_depth': [3, 4, 5, 6, 7, 8, 9],
        'min_child_weight': [1, 2, 3, 4, 5],
        'gamma': [i/10.0 for i in range(0, 4)],
        'subsample': [i/10.0 for i in range(6, 9)],
        'colsample_bytree': [i/10.0 for i in range(6, 9)],
        'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 1]
    }
    print('Starting GridSearchCV')
    gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,
                                                   n_estimators=140,
                                                   max_depth=5,
                                                   min_child_weight=1,
                                                   gamma=0,
                                                   subsample=0.8,
                                                   colsample_bytree=0.8,
                                                   objective='binary:logistic',
                                                   scale_pos_weight=1,
                                                   seed=24),
                            param_grid=parameters,
                            scoring='roc_auc',
                            iid=True,
                            cv=5)
    gsearch.fit(result_list[0], result_list[2])
    print('Done with GridSearchCV')
    gscv_results = [gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_]
    name = 'Results/GSCV/GSCV_resutlts.pkl'
    pickle.dump(gscv_results, open(name, 'wb'))

if __name__ == '__main__':
    main()

