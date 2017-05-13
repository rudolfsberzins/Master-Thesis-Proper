import sys
sys.path.append('../../Core-scripts')
import file_readers as fr
import pickle
import logging
import glob
import numpy as np
from nltk.tokenize import sent_tokenize
from gensim.models import word2vec

def sentence_yielder():
    for file in glob.glob('Results/full_PubMed_dict_*.pkl'):
        dict_file = pickle.load(open(file, 'rb'))
        for sen in dict_file:
            yield fr.sentence_to_wordlist(sen)

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

    num_features = 600  # Word vector dimensionality
    min_word_count = 6  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 6  # Context window size
    downsampling = 0.000001  # Downsample setting for frequent words

    print('Training Word2Vec Model')

    model = word2vec.Word2Vec([i for i in sentence_yielder()], workers=num_workers, \
            size=num_features, min_count=min_word_count, \
            window=context, sample=downsampling)

    model.init_sims(replace=True)

    model_name = 'Results/full_PubMed_word2vec_model'

    model.save(model_name)

if __name__ == '__main__':
    main()

