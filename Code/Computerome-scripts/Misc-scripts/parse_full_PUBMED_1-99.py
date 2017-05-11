import glob
import gzip
import csv
import os
import sys
import pickle
from nltk.tokenize import sent_tokenize

maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

def yield_corpus_names(med_folder):
    """Produces MEDLINE abstract gzip filenames"""

    for i in range(1, 100):
        file_path = 'medline16n'+str(i).zfill(4)+'.tsv.gz'
        for filename in glob.glob(os.path.join(med_folder, file_path)):
            yield filename

def yield_abstract_corpus(med_fold):
    """Yields a single abstract line"""
    for path in yield_corpus_names(med_fold):
        with gzip.open(path, 'rt', encoding='utf-8') as corpus:
            tsv_reader = csv.reader(corpus, delimiter='\t')
            for row in tsv_reader:
                yield row

def main():

    if not os.path.exists('Results'):
        os.makedirs('Results')

    medline_folder = '../../../../MEDLINE_FILES'
    full_mention_dict = pickle.load(open('Results/full_mention_dict.pkl', 'rb'))

    full_PubMed_dict = {}
    for row in yield_abstract_corpus(medline_folder):
        try:
            text_of_interest = row[4] + row[5]
            for sentence in sent_tokenize(text_of_interest):
                word_list = sentence.split()
                word_list_clean = [word for word in word_list if word not in full_mention_dict]
                word_list_clean = ' '.join(word_list_clean)
                full_PubMed_dict[word_list_clean] = list(set(word_list) - set(word_list_clean))

        except IndexError:
            print ('Got Index Error with: ', row[0], '! Passing')

    pickle.dump(full_PubMed_dict, open('Results/full_PubMed_dict_1-99.pkl', 'wb'))

if __name__ == '__main__':
    main()

