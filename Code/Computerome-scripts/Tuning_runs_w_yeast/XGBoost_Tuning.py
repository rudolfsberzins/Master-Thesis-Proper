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
    _, seed, model = argv

    result_list_name = 'Results/yeast_strict_list_'+str(model)+'_'+str(seed)+'_results_list.pkl'
    result_list = pickle.load(open(result_list_name, 'rb'))

    parameters = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'min_child_weight': [1, 2, 3, 4, 5, 6],
        'gamma': [i/10.0 for i in range(0, 5)],
        'subsample': [i/10.0 for i in range(6,10)],
        'colsample_bytree': [i/10.0 for i in range(6,10)],
        'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 1, 100]
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
                            iid=False,
                            cv=5)
    gsearch.fit(result_list[0], result_list[2])
    print('Done with GridSearchCV')
    gscv_results = [gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_]
    name = 'Results/GSCV_'+str(model)+'_'+str(seed)+'_resutlts.pkl'
    pickle.dump(gscv_results, open(name, 'wb'))

if __name__ == '__main__':
    main()

