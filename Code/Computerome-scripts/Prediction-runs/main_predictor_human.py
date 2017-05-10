import pickle
from parse_and_prepare import ProteinProteinInteractionClassifier
import prediction as pred
from sklearn.metrics import roc_curve
import numpy as np

def main():
    files = ['MEDLINE_FILES',
             'human_mentions',
             'human_entities.tsv',
             '9606.protein.actions.v10.txt']

    # Preloaded Pickles
    dros_strict = pickle.load(open('Results/strict_pairs_w_sen_df.pkl', 'rb'))
    dros_gen = pickle.load(open('Results/gen_pairs_w_sen_df.pkl', 'rb'))
    dros_both_ents = pickle.load(open('Results/both_ents_w_sen_df.pkl', 'rb'))

    dros_strict_real = pickle.load(open('Results/strict_real.pkl', 'rb'))
    dros_gen_real = pickle.load(open('Results/gen_real.pkl', 'rb'))
    dros_be_real = pickle.load(open('Results/be_real.pkl', 'rb'))

    dros_pickle_list = [dros_strict, dros_strict_real,
                        dros_gen, dros_gen_real,
                        dros_both_ents, dros_be_real]

    np.random.seed(1)
    random_seeds = np.random.randint(9999, size=10)

    print(random_seeds)

    print ('\nLoading and Preping data \n')
    clf = ProteinProteinInteractionClassifier(files, pre_loaded_files=dros_pickle_list, \
        full_sen_set=True)
    # clf = ProteinProteinInteractionClassifier(files, full_sen_set=True)
    # clf.prepare()
    # clf.produce_pairs()
    # clf.add_real_ids_and_mask()
    print ('\nMaking Models and Feature Vectors\n')
    gen_model = pred.make_w2v_model(clf.gen_real, 'human_gen_real')
    both_ents_model = pred.make_w2v_model(clf.be_real, 'human_both_ents')
    strict_model = pred.make_w2v_model(clf.strict_real, 'human_strict_real')

    for seed in random_seeds:
        strict_list_pure = pred.make_models(clf.strict_real, 'human_strict_real_pure_'+str(seed),
                                            prev_model=strict_model, ran_state=seed)
        strict_list_gen_mod = pred.make_models(clf.strict_real,
                                               'human_strict_real_gen_mod_'+str(seed),
                                               prev_model=gen_model, ran_state=seed)
        strict_list_be_mod = pred.make_models(clf.strict_real,
                                              'human_strict_real_be_mod_'+str(seed),
                                              prev_model=both_ents_model, ran_state=seed)

        strict_final_list = [strict_list_pure, strict_list_gen_mod, strict_list_be_mod]

        print ('\nPredicting\n')
        errors = []
        fpr = []
        tpr = []

        for entry in strict_final_list:
            error_w2v_norm, probs_w2v_norm = pred.XGB_classifier(entry[0], entry[1],
                                                                 entry[2], entry[3])
            fpr_w2v_norm, tpr_w2v_norm, _ = roc_curve(entry[3], probs_w2v_norm)
            error_w2v_fs, probs_w2v_fs = pred.XGB_classifier(entry[0], entry[1],
                                                             entry[2], entry[3],
                                                             feature_selection=True)
            fpr_w2v_fs, tpr_w2v_fs, _ = roc_curve(entry[3], probs_w2v_fs)

            errors.append([error_w2v_norm, error_w2v_fs])
            fpr.append([fpr_w2v_norm, fpr_w2v_fs])
            tpr.append([tpr_w2v_norm, tpr_w2v_fs])

        pickle.dump(errors, open('Results/human_errors_pickle_'+str(seed)+'.pkl', 'wb'))
        pickle.dump(fpr, open('Results/human_fpr_pickle_'+str(seed)+'.pkl', 'wb'))
        pickle.dump(tpr, open('Results/human_tpr_pickle_'+str(seed)+'.pkl', 'wb'))



if __name__ == '__main__':
    main()
