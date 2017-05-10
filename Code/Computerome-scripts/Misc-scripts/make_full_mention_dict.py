import pickle
import os
import glob
import pandas as pd


def main():

    if not os.path.exists('Results'):
        os.makedirs('Results')

    full_mention_dict = {}

    for file in glob.glob('/data/*_mentions'):
        mentions_file = pd.read_csv(file, sep='\t',
                                    names=["PMID", "Paragraph", "Sentence", "Char_Start",
                                           "Char_End", "Entity_name", "TaxID", "SerialNo"])
        all_names = set(mentions_file['Entity_name'])

        for name in all_names:
            full_mention_dict[name] = ''

    pickle.dump(full_mention_dict, open('Results/full_mention_dict.pkl', 'wb'))

if __name__ == '__main__':
    main()



