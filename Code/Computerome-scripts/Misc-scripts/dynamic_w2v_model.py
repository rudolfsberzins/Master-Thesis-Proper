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

def main():

    _, sentence_path, name_model = argv

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

    num_features = 600  # Word vector dimensionality
    min_word_count = 6  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 7  # Context window size
    downsampling = 0.0001  # Downsample setting for frequent words

    print('Training Word2Vec Model')

    model = word2vec.Word2Vec([i for i in sentence_yielder(sentence_path)], workers=num_workers,
                              size=num_features, min_count=min_word_count, \
                              window=context, sample=downsampling)

    model.init_sims(replace=True)

    model_name = 'Results/' + name_model

    model.save(model_name)

if __name__ == '__main__':
    main()

