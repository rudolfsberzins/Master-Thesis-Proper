from sys import argv
import file_readers as fr
import pickle
import logging
import glob
import numpy as np
from nltk.tokenize import sent_tokenize
from gensim.models import word2vec
import itertools

def sentence_yielder(path):
    for file in glob.glob(path):
        dict_file = pickle.load(open(file, 'rb'))
        for sen in dict_file:
            yield fr.sentence_to_wordlist(sen)

def full_merger(model_path, sen_pkl_path, model_name):

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

    #Load previous model
    model = word2vec.Word2Vec.load(model_path)

    #Build a new vocab
    model.build_vocab([i for i in sentence_yielder(sen_pkl_path)], update=True)

    #Train model using new sentences
    model.train([i for i in sentence_yielder(sen_pkl_path)])

    #'Kill' the model
    model.init_sims(replace=False)

    model_name = 'Results/models/' + model_name
    model.save(model_name)

    print('Done with ', model_name)

def make_mergers(name_start, name_target, init_trigger=False, human_trigger=False):
    if init_trigger:
        path_to_strict_model = 'Results/models/'+name_start+'__strict_w2v_model'
        path_to_gen_model = 'Results/models/'+name_start+'__gen_w2v_model'
        path_to_be_model = 'Results/models/'+name_start+'__be_w2v_model'
    else:
        path_to_strict_model = 'Results/models/'+name_start+'_full_merger_SR_model'
        path_to_gen_model = 'Results/models/'+name_start+'_full_merger_GEN_model'
        path_to_be_model = 'Results/models/'+name_start+'_full_merger_BE_model'

    if human_trigger:
        path_to_strict_data = '../'+name_target+'/computerome_human_runs/Results/'+name_target+'_mentions_strict_real.pkl'
        path_to_gen_data = '../'+name_target+'/computerome_human_runs/Results/'+name_target+'_mentions_gen_real.pkl'
        path_to_be_data = '../'+name_target+'/computerome_human_runs/Results/'+name_target+'_mentions_be_real.pkl'
    else:
        path_to_strict_data = '../'+name_target+'/Results/'+name_target+'mentions_strict_real.pkl'
        path_to_gen_data = '../'+name_target+'/Results/'+name_target+'mentions_gen_real.pkl'
        path_to_be_data = '../'+name_target+'/Results/'+name_target+'mentions_be_real.pkl'

    name_of_results = name_start+'_'+name_target

    full_merger(path_to_strict_model,
                path_to_strict_data,
                name_of_results+'_full_merger_SR_model')

    full_merger(path_to_gen_model,
                path_to_gen_data,
                name_of_results+'_full_merger_GEN_model')

    full_merger(path_to_be_model,
                path_to_be_data,
                name_of_results+'_full_merger_BE_model')

def iterator(lst):
    for i in lst:
        yield i

def main():
    _, folder_name, target_1, target_2, target_3, target_4 = argv
    names = [folder_name, target_1, target_2, target_3, target_4]
    names_true = []
    for i in names[1:]:
        if i !='BLANK':
            names_true.append(i)

    i_trigger = False
    h_trigger = False

    all_names = []
    for i in range(1, len(names_true)+1):
        z = itertools.combinations(iterator(names_true), i)
        for j in z:
            d = '_'.join(list(j))
            all_names.append(d)

    for single_name in all_names:
        splt = single_name.split('_', 1)
        name_target = splt[0]
        if name_target == 'human':
            h_trigger = True
        if len(splt) == 1:
            name_start = names[0]
            i_trigger = True
        else:
            name_start = names[0]+'_'+splt[1]

        make_mergers(name_start, name_target, i_trigger, h_trigger)

        i_trigger = False
        h_trigger = False



if __name__ == '__main__':
    main()
