import pickle
import prediction as pred
from sys import argv
from gensim import word2vec
from sklearn.metrics import roc_curve, auc

def main():
    _, path_to_strict_model, path_to_gen_model, path_to_be_model, path_to_strict_data, name_of_result = argv
    random_seeds = [144, 235, 905, 2895, 3462, 4225, 5056, 5192, 7751, 7813]

    w2v_strict = word2vec.Word2Vec.load(path_to_strict_model)
    w2v_gen = word2vec.Word2Vec.load(path_to_gen_model)
    w2v_be = word2vec.Word2Vec.load(path_to_be_model)

    strict_data = pickle.load(open(path_to_strict_data, 'rb'))

    for seed in random_seeds:
        strict_list_SR = pred.make_models(strict_data,
                                          name_of_result+'_SR_'+str(seed),
                                          prev_model=w2v_strict,
                                          ran_state=seed)

        strict_list_GEN = pred.make_models(strict_data,
                                           name_of_result+'_GEN_'+str(seed),
                                           prev_model=w2v_gen,
                                           ran_state=seed)
        strict_list_BE = pred.make_models(strict_data,
                                          name_of_result+'_BE_'+str(seed),
                                          prev_model=w2v_gen,
                                          ran_state=seed)

        strict_final_list = [strict_list_SR,
                             strict_list_GEN,
                             strict_list_BE]

        print ('\nPredicting\n')
        errors = []
        probs = []
        fpr = []
        tpr = []
        labels = []

        for entry in strict_final_list:
            pred_labels_norm, error_w2v_norm, probs_w2v_norm = pred.XGB_classifier(entry[0],
                                                                                   entry[1],
                                                                                   entry[2],
                                                                                   entry[3])
            fpr_w2v_norm, tpr_w2v_norm, _ = roc_curve(entry[3], probs_w2v_norm)
            pred_labels_fs, error_w2v_fs, probs_w2v_fs = pred.XGB_classifier(entry[0], entry[1],
                                                                             entry[2], entry[3],
                                                                             feature_selection=True)
            fpr_w2v_fs, tpr_w2v_fs, _ = roc_curve(entry[3], probs_w2v_fs)

            errors.append([error_w2v_norm, error_w2v_fs])
            probs.append([probs_w2v_norm, probs_w2v_fs])
            fpr.append([fpr_w2v_norm, fpr_w2v_fs])
            tpr.append([tpr_w2v_norm, tpr_w2v_fs])
            labels.append([pred_labels_norm, pred_labels_fs])

        pickle.dump(errors, open('Results/'+name_of_result+'_errors_pickle_'+str(seed)+'.pkl',
                                 'wb'))
        pickle.dump(probs, open('Results/'+name_of_result+'_probs_pickle_'+str(seed)+'.pkl',
                                'wb'))
        pickle.dump(fpr, open('Results/'+name_of_result+'_fpr_pickle_'+str(seed)+'.pkl',
                              'wb'))
        pickle.dump(tpr, open('Results/'+name_of_result+'_tpr_pickle_'+str(seed)+'.pkl',
                              'wb'))
        pickle.dump(labels, open('Results/'+name_of_result+'_labels_pickle_'+str(seed)+'.pkl',
                                 'wb'))


if __name__ == '__main__':
    main()

