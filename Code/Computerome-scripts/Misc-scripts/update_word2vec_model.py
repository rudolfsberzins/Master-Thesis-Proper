from sys import argv
import file_readers as fr
import pickle
import logging
import glob
import numpy as np
from nltk.tokenize import sent_tokenize
from gensim.models import word2vec

def sentence_yielder(path):
    for file in glob.glob(path):
        dict_file = pickle.load(open(file, 'rb'))
        for sen in dict_file:
            yield fr.sentence_to_wordlist(sen)

def add_sentences_to_model(model_path, sen_pkl_path, model_name):

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

    #Load previous model
    model = word2vec.Word2Vec.load(model_path)

    #Build a new vocab
    model.build_vocab([i for i in sentence_yielder(sen_pkl_path)], update=True)

    #Train model using new sentences
    model.train([i for i in sentence_yielder(sen_pkl_path)])

    #'Kill' the model
    model.init_sims(replace=True)

    model_name = 'Results/' + model_name
    model.save(model_name)

    print('Done with ', model_name)

def main():

    _, model_path, sentence_path, model_name = argv

    add_sentences_to_model(model_path,
                           sentence_path, model_name)

if __name__ == '__main__':
    main()
