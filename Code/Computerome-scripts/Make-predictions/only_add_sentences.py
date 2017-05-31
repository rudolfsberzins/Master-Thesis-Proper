import pickle
import prediction as pred
import file_readers as fr
from sys import argv
from gensim import word2vec
import logging

def add_sentences_to_model(model_path, sen_pkl_path, model_name):

    #Prepare datasets
    pickle_dataset = pickle.load(open(sen_pkl_path, 'rb'))
    sentences = [fr.sentence_to_wordlist(sen) for sen in pickle_dataset]

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

    #Load previous model
    model = word2vec.Word2Vec.load(model_path)

    #Train model using new sentences
    model.train(sentences)

    #'Kill' the model
    model.init_sims(replace=False)

    model_name = 'Results/' + model_name
    model.save(model_name)
    w2v_model = model

    return w2v_model

def main():
    _, path_to_strict_model, path_to_strict_data, path_to_gen_model, path_to_gen_data, path_to_be_model, path_to_be_data, name_of_results = argv

    add_sentences_to_model(path_to_strict_model,
                           path_to_strict_data,
                           name_of_results+'_SR_model')

    add_sentences_to_model(path_to_gen_model,
                           path_to_gen_data,
                           name_of_results+'_GEN_model')

    add_sentences_to_model(path_to_be_model,
                           path_to_be_data,
                           name_of_results+'_BE_model')

if __name__ == '__main__':
    main()
