import sys
sys.path.insert(0, '../../Core-scripts/')

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
import multiprocessing

def main():
    _, seed, model = argv


    result_list_name = 'Results/result_list/yeast_'+str(model)+'_'+str(seed)+'_no_BOW_results_list.pkl'
    result_list = pickle.load(open(result_list_name, 'rb'))

    parameters = {
        'max_depth': [1, 2, 3, 4, 5, 6],
        'min_child_weight': [1, 2, 3, 4, 5],
        'gamma': [i/10.0 for i in range(0, 5)],
        'subsample': [i/10.0 for i in range(6, 10)],
        'colsample_bytree': [i/10.0 for i in range(6, 10)],
    }
    print('Starting GridSearchCV')
    gsearch = GridSearchCV(estimator=XGBClassifier(seed=24),
                           param_grid=parameters,
                           scoring='roc_auc',
                           iid=True,
                           n_jobs=multiprocessing.cpu_count(),
                           cv=3)
    gsearch.fit(result_list[0], result_list[2])
    print('Done with GridSearchCV')
    gscv_results = [gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_]
    name = 'Results/GSCV/GSCV_'+str(model)+'_'+str(seed)+'_resutlts.pkl'
    pickle.dump(gscv_results, open(name, 'wb'))

if __name__ == '__main__':
    main()

