import re
import glob
import gzip
import csv
import os
import time
import sys
import pickle
import logging
import file_readers as fr
import pandas as pd
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

class ProteinProteinInteractionClassifier(object):
    def __init__(self, files, pre_loaded_files=None, full_sen_set=False):
        """
        Opens files provided in a list with the following order -
        0 - Path to folder containing MEDLINE abstracts,
        1 - Mentions file from Text-mining tagger,
        2 - Pairs file from Text-mining tagger,
        3 - ORGANISM entities file from ORGANISM dictionary,
        4 - Protein interactions file for ORGANISM from STRING
        """

        # Define paths to all files
        container = 'data/'
        self.medline_folder = '../../../../Medline'
        # self.medline_folder = os.path.join(container, files[0])
        self.mentions_path = os.path.join(container, files[1])
        self.entities_path = os.path.join(container, files[2])
        self.interactions_path = os.path.join(container, files[3])

        #Make results folder if it doesn't exist
        if not os.path.exists('Results'):
            os.makedirs('Results')

        # Dictates if make additional pairs dataframe with full sentence set (not true pairs)
        self.full_sen_set = full_sen_set

        # Initiate variables
        self.mentions = None
        self.ents = None
        self.interactions = None

        self.res_name = files[1]

        self.strict_pairs_w_sen_df = None
        self.strict_real = None
        if self.full_sen_set:
            self.gen_pairs_w_sen_df = None
            self.gen_real = None
            self.both_ents_w_sen_df = None
            self.be_real = None

        if pre_loaded_files:
            self.strict_pairs_w_sen_df = pre_loaded_files[0]
            self.strict_real = pre_loaded_files[1]
            if self.full_sen_set:
                self.gen_pairs_w_sen_df = pre_loaded_files[2]
                self.gen_real = pre_loaded_files[3]
                self.both_ents_w_sen_df = pre_loaded_files[4]
                self.be_real = pre_loaded_files[5]


    def prepare(self):
        """Open and prepare files"""
        self.mentions = fr.open_mentions_and_pairs(self.mentions_path)
        self.ents = fr.open_entities(self.entities_path)
        self.interactions = fr.open_interactions(self.interactions_path)


    def yield_corpus_names(self):
        """Produces MEDLINE abstract gzip filenames"""
        for filename in glob.glob(os.path.join(self.medline_folder, '*.tsv.gz')):
            yield filename

    def yield_abstract_corpus(self):
        """Yields a single abstract line"""
        for path in self.yield_corpus_names():
            with gzip.open(path, 'rt', encoding='utf-8') as corpus:
                tsv_reader = csv.reader(corpus, delimiter='\t')
                for row in tsv_reader:
                    yield row

    def find_common(self):
        """Finds the common PubMed ID's between the MEDLINE abstacts and mentions file

        INPUT: -

        OUTPUT: List of articles (PMID's) both DF's share"""

        journals = set()
        for row in self.yield_abstract_corpus():
            hit = re.findall('PMID:\d*', row[0])[0][5:]
            journals.add(int(hit))
        common = tuple(set(self.mentions['PMID']) & journals)

        return common

    def extract_true_pairs(self, both_entries):
        """Finds true pairs, meaning those who are in the same paragraph, same sentence,
        but have different serial ID's and entity names

        INPUT: Data Frame which consists of entries where both genes are in the same text

        OUTPUT: Data Frame of True Pairs"""

        column_titles = ["PMID", "Paragraph", "Sentence", "Char_Start_x", "Char_End_x",
                         "Entity_name_x", "TaxID", "SerialNo_x", "Char_Start_y", "Char_End_y",
                         "Entity_name_y", "SerialNo_y"]

        results = pd.DataFrame(columns=column_titles)
        print ('\n')
        print (both_entries.shape)
        # To Break out of Loop and to prevent StopIteration error
        temp_df = pd.DataFrame([['STOP', 0, 0, 0, 0, '', 0, 0]], columns=["PMID", "Paragraph", "Sentence", "Char_Start", "Char_End", "Entity_name", "TaxID", "SerialNo"])
        both_entries = both_entries.append(temp_df, ignore_index=True)
        print (both_entries.shape)
        print ('\n')

        row_iterator = both_entries.iterrows()
        _, last = row_iterator.__next__()
        for idx, row in row_iterator:
            if row["PMID"] == last["PMID"] and row["Paragraph"] == last["Paragraph"] and row["Sentence"] == last["Sentence"] and row["Entity_name"] != last["Entity_name"] and row["SerialNo"] != last["SerialNo"]:

                row_df = pd.DataFrame(row).T #I know this is dirty, but it works
                last_df = pd.DataFrame(last).T #I know this is dirty, but it works
                merger = pd.merge(row_df, last_df, on=["PMID", "Paragraph", "Sentence", "TaxID"],
                                  how="inner")
                results = results.append(merger, ignore_index=True)
                last = row
                if row['PMID'] == 'STOP':
                    print('broke')
                    print(idx)
                    break
            else:
                last = row
                if row['PMID'] == 'STOP':
                    print('BROKE')
                    print(idx)
                    break

        results[["PMID", "Paragraph", "Sentence", "Char_Start_x",
                 "Char_End_x", "TaxID", "SerialNo_x", "Char_Start_y",
                 "Char_End_y", "SerialNo_y"]] = results[["PMID", "Paragraph",
                                                         "Sentence", "Char_Start_x",
                                                         "Char_End_x", "TaxID", "SerialNo_x",
                                                         "Char_Start_y", "Char_End_y",
                                                         "SerialNo_y"]].astype(int)

        return results

    def add_sentences(self, pairs_file, true_check, pairs_check):
        """Adds sentences for true pairs

        INPUT: Pairs DF and whether or not it is True Pairs DF (Boolean)

        OUTPUT: Pairs DF with added sentences"""

        pairs_file['Text'] = None
        pmid_dict = {key: '' for key in pairs_file['PMID'].tolist()}
        for row in self.yield_abstract_corpus():
            hit = re.findall('PMID:\d*', row[0])[0][5:]
            if int(hit) in pmid_dict:
                try:
                    text_of_interest = row[4] + row[5]
                    mapping = self.get_b2c_mapping(text_of_interest)

                    try:
                        temp_hits = pairs_file.loc[(pairs_file['PMID'] == int(hit)),
                                                   ['Char_Start_x']]
                        mapped_temp_hits = []
                        for index, single_hit in temp_hits.iterrows():
                            try:
                                single_hit = mapping[int(single_hit['Char_Start_x'])]
                            except KeyError:
                                single_hit = int(single_hit['Char_Start_x'])
                            mapped_temp_hits.append((index, int(single_hit)))
                    except KeyError:
                        temp_hits = pairs_file.loc[(pairs_file['PMID'] == int(hit)),
                                                   ['Char_Start']]
                        mapped_temp_hits = []
                        for index, single_hit in temp_hits.iterrows():
                            try:
                                single_hit = mapping[int(single_hit['Char_Start'])]
                            except KeyError:
                                single_hit = int(single_hit['Char_Start'])
                            mapped_temp_hits.append((index, int(single_hit)))

                    if true_check:
                        if pairs_check and len(mapped_temp_hits) == 1:
                            sublen = 0
                            for sentence in sent_tokenize(text_of_interest):
                                for ind, line in mapped_temp_hits:
                                    if 0 < line - sublen < len(sentence):
                                    # -49 (in Python 3.x, -37 in Python 2.x) because that is the
                                    # difference between len(x) and sys.getsizeof(x)
                                        pairs_file.set_value(ind, 'Text', sentence)
                                        break #Break to catch on the first sentence
                                sublen += len(sentence)
                        elif not pairs_check:
                            sublen = 0
                            for sentence in sent_tokenize(text_of_interest):
                                for ind, line in mapped_temp_hits:
                                    if 0 < line - sublen < len(sentence):
                                    # -49 (in Python 3.x, -37 in Python 2.x) because that is the
                                    # difference between len(x) and sys.getsizeof(x)
                                        pairs_file.set_value(ind, 'Text', sentence)
                                sublen += len(sentence)
                    elif not true_check:
                        sublen = 0
                        for sentence in sent_tokenize(text_of_interest):
                            for ind, line in mapped_temp_hits:
                                if 0 < line - sublen < len(sentence):
                                # -49 (in Python 3.x, -37 in Python 2.x) because that is the
                                # difference between len(x) and sys.getsizeof(x)

                                    pairs_file.set_value(ind, 'Text', sentence)
                            sublen += len(sentence)
                except IndexError:
                    print ('Got Index Error with: ' + str(hit) + '! Passing')
        pairs_file = pairs_file.dropna(axis=0, how='any')
        return pairs_file

    def get_b2c_mapping(self, document):
        # get byte to character mapping for document
        # for each byte in the document, determine if it belongs to a multi byte character
        mapping = {} # byte to char
        byte = 0
        character = 0
        #u_document = document.decode("utf-8") # turn bytes into characters
        for b in document:
            u = b.encode('utf-8') # back to bytes
            char_bytes = len(u) # how many bytes does this character consist of
            for i in range(0, char_bytes):
                mapping[byte+i] = character
            byte += char_bytes
            character += 1
        return mapping



    def produce_pairs(self):
        """
        Produces 3 types of Pairs Dataframes depening on parametrs

        1. self.strict_pairs_df_w_sen - Contains only True pairs and
           sentences with only 2 proteins in them
        2. self.gen_pairs_df_w_sen - Contans only True pairs,
           but a sentence can have more than 2 proteins
        3. self.both_ents_w_sen_df - Contains all proteins where both they
           and their pair are in MEDLINE files

        Both ents should be used only as unsupervised dataset

        """

        # print('\n Producing Pairs DFs \n')
        # common_j = self.find_common()

        # print ('length of common j', len(common_j))

        # #This gives all articles that are in mentions and available in MEDLINE files
        # shared_entities = self.mentions[self.mentions['PMID'].isin(common_j)]
        # shared_entities = shared_entities.sort_values('SerialNo')

        # print('shared entities', shared_entities.shape)

        # #This gives DF where there are atleast two entries in them
        # ser_no_ents = shared_entities['SerialNo'].tolist()
        # temp_s1_values = self.pairs[self.pairs['SerialNo1'].isin(ser_no_ents)]
        # temp_s2_list = temp_s1_values['SerialNo2'].tolist()
        # both_entries = shared_entities[shared_entities['SerialNo'].isin(temp_s2_list)]
        # both_entries = both_entries.sort_values('PMID')

        # print('ser_no_ents', len(ser_no_ents))
        # print('temp_s1_values', temp_s1_values.shape)
        # print('temp_s2_list', len(temp_s2_list))
        # print('both_entries', both_entries.shape)

        both_entries = self.mentions.sort_values('PMID')

        #Get True Pairs
        true_pairs = self.extract_true_pairs(both_entries)


        tp_strict_with_sen = self.add_sentences(true_pairs, True, True)

        print('Strict shape before drop, ', tp_strict_with_sen.shape)
        tp_strict_with_sen = tp_strict_with_sen[pd.notnull(tp_strict_with_sen['Text'])]
        print('Strict shape after drop, ', tp_strict_with_sen.shape)


        tp_strict_with_sen[['SerialNo_x',
                            'SerialNo_y']] = tp_strict_with_sen[['SerialNo_x',
                                                                 'SerialNo_y']].astype(int)
        self.strict_pairs_w_sen_df = tp_strict_with_sen[['Entity_name_x',
                                                         'SerialNo_x', 'Entity_name_y',
                                                         'SerialNo_y', 'Text']]

        name_strict = 'Results/' + self.res_name + '_strict_pairs_w_sen_df.pkl'

        pickle.dump(self.strict_pairs_w_sen_df, open(name_strict, 'wb'))

        if self.full_sen_set:
            tp_gen_with_sen = self.add_sentences(true_pairs, True, False)
            both_entries_with_sen = self.add_sentences(both_entries, False, False)

            print('Gen shape before drop, ', tp_gen_with_sen.shape)
            print('Both ents shape before drop, ', both_entries_with_sen.shape)
            tp_gen_with_sen = tp_gen_with_sen[pd.notnull(tp_gen_with_sen['Text'])]
            both_entries_with_sen = both_entries_with_sen[pd.notnull(both_entries_with_sen['Text'])]
            print('Gen shape after drop, ', tp_gen_with_sen.shape)
            print('Both ents shape after drop, ', both_entries_with_sen.shape)



            tp_gen_with_sen[['SerialNo_x',
                             'SerialNo_y']] = tp_gen_with_sen[['SerialNo_x',
                                                               'SerialNo_y']].astype(int)

            both_entries_with_sen[['SerialNo']] = both_entries_with_sen[['SerialNo']].astype(int)

            self.gen_pairs_w_sen_df = tp_gen_with_sen[['Entity_name_x',
                                                       'SerialNo_x', 'Entity_name_y',
                                                       'SerialNo_y', 'Text']]
            self.both_ents_w_sen_df = both_entries_with_sen[['Entity_name',
                                                             'SerialNo', 'Text']]

            name_gen = 'Results/' + self.res_name + '_gen_pairs_w_sen_df.pkl'
            name_be = 'Results/' + self.res_name + '_both_ents_w_sen_df.pkl'

            pickle.dump(self.gen_pairs_w_sen_df, open(name_gen, 'wb'))
            pickle.dump(self.both_ents_w_sen_df, open(name_be, 'wb'))

        print('\n Done with Pairs DFs \n')


    def add_real_ids_and_mask(self):
        """
        Makes dictionaries with real ids from dataframes:

        Final datastructure: {'Text': [['Entity Name', Real_ID']]
        """
        dropped = 0
        print('\n Adding Real_IDs in to the mix \n')
        given_ents_dict = self.ents
        self.strict_real = fr.df_to_dict(self.strict_pairs_w_sen_df)

        for text, lists in self.strict_real.copy().items():
            for_subbing = text
            for single_list in lists:
                if single_list[1] in given_ents_dict and single_list[3] in given_ents_dict:
                    single_list[1] = self.ents[single_list[1]][0]
                    single_list[3] = self.ents[single_list[3]][0]
                sub1 = for_subbing.replace(single_list[0], '')
                sub2 = sub1.replace(single_list[2], '')
                for_subbing = sub2

            if for_subbing == text:
                dropped += 1
                del self.strict_real[text]
            else:
                self.strict_real[for_subbing] = self.strict_real.pop(text)


        print(' * Adding interactions to Strict Pairs ')
        self.strict_real = self.add_interactions(self.strict_real)

        name_s_real = 'Results/' + self.res_name + '_strict_real.pkl'

        pickle.dump(self.strict_real, open(name_s_real, 'wb'))

        if self.full_sen_set:
            self.gen_real = fr.df_to_dict(self.gen_pairs_w_sen_df)

            for text, lists in self.gen_real.copy().items():
                for_subbing = text #This allows to make multiple changes to the key and then pop it
                for single_list in lists:
                    if single_list[1] in given_ents_dict and single_list[3] in given_ents_dict:
                        single_list[1] = self.ents[single_list[1]][0]
                        single_list[3] = self.ents[single_list[3]][0]
                    sub1 = for_subbing.replace(single_list[0], '')
                    sub2 = sub1.replace(single_list[2], '')
                    for_subbing = sub2

                if for_subbing == text:
                    dropped += 1
                    del self.gen_real[text]
                else:
                    self.gen_real[for_subbing] = self.gen_real.pop(text)

            # print(' * Adding interactions to General Pairs ')
            # self.gen_real = self.add_interactions(self.gen_real)

            name_g_real = 'Results/' + self.res_name + '_gen_real.pkl'

            pickle.dump(self.gen_real, open(name_g_real, 'wb'))

            self.be_real = fr.df_to_dict(self.both_ents_w_sen_df, True)

            for text, lists in self.be_real.copy().items():
                for_subbing = text #This allows to make multiple changes to the key and then pop it
                for single_list in lists:
                    if single_list[1] in given_ents_dict:
                        single_list[1] = self.ents[single_list[1]][0]
                    sub = for_subbing.replace(single_list[0], '')
                    for_subbing = sub

                if for_subbing == text:
                    dropped += 1
                    del self.be_real[text]
                else:
                    self.be_real[for_subbing] = self.be_real.pop(text)

            name_be_real = 'Results/' + self.res_name + '_be_real.pkl'

            pickle.dump(self.be_real, open(name_be_real, 'wb'))

        print('\n', str(dropped), ' texts were dropped')

        print('\n Done with Real_IDs \n')

    def add_interactions(self, nec_dict):
        """Add interactions to the dictionaries!"""

        desired_interaction = 'binding'

        drop_keys = []
        for key, value in nec_dict.items():
            for single_list in value:
                if single_list[1] in self.interactions:
                    pre_mode = self.interactions[single_list[1]]
                    if single_list[3] in pre_mode:
                        if desired_interaction in pre_mode[single_list[3]]:
                            single_list.append(1)
                        else:
                            single_list.append(0)
                    else:
                        single_list.append(0)
                else:
                    drop_keys.append(key)

        filtered_dict = {}
        for key, value in nec_dict.items():
            if key not in drop_keys:
                filtered_dict[key] = value

        return filtered_dict




