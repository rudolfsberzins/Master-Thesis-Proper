import re
import pandas as pd
from nltk.corpus import stopwords



def open_mentions_and_pairs(men_name):
    """Opens the mentions file and pairs file from tagger made by JensenGroup from NNCPR

    INPUT: mentions file (.tsv), pairs file (.tsv)

    OUTPUT: Pandas DataFrame of the corresponding .tsv files"""

    men_file = pd.read_csv(men_name,
                           sep='\t',
                           names=["PMID", "Paragraph", "Sentence", "Char_Start",
                                  "Char_End", "Entity_name", "TaxID", "SerialNo"])
    # pairs_file = pd.read_csv(pairs_name,
    #                          sep='\t',
    #                          names=["SerialNo1", "SerialNo2", "Final_Score",
    #                                 "UNUSED1", "UNUSED2", "UNUSED3", "UNUSED4",
    #                                 "UNUSED5"])

    return men_file

def open_entities(ents_name):
    """Opens the mentions file and pairs file from tagger made by JensenGroup from NNCPR

    INPUT: mentions file (.tsv), pairs file (.tsv)

    OUTPUT: Pandas DataFrame of the corresponding .tsv files"""

    ents = pd.read_csv(ents_name,
                       sep='\t',
                       names=['Given_ID', 'UNUSED', 'Real_ID'])

    ents_dict = {}

    for _, row in ents.iterrows():
        ents_dict.setdefault(row['Given_ID'], []).append(row['Real_ID'])

    return ents_dict

def df_to_dict(dataframe, be_frame=False):
    """
    Transform a dataframe to the following structure:

        {'Text': [('Entity_name', SerialNo)]}

    Argument be_frame (Bool) designates the Both_Entries Dataframe
    """
    trans_dict = {}

    for _, row in dataframe.iterrows():
        if not be_frame:
            trans_dict.setdefault(row['Text'], []).append([row['Entity_name_x'],
                                                           row['SerialNo_x'],
                                                           row['Entity_name_y'],
                                                           row['SerialNo_y']])
        else:
            trans_dict.setdefault(row['Text'], []).append([row['Entity_name'],
                                                           row['SerialNo']])

    return trans_dict

def open_interactions(interactions_name):
    """Opens a TSV File Containing the Interactions from STRING DB

    INPUT: Name of the tsv file

    OUTPUT: Pandas DataFrame of the .tsv file"""

    interactions_file = pd.read_csv(interactions_name, sep='\t')

    interactions_file['item_id_a'] = interactions_file['item_id_a'].str.split('.').apply(lambda x: x[1])
    interactions_file['item_id_b'] = interactions_file['item_id_b'].str.split('.').apply(lambda x: x[1])
    interactions_file = interactions_file[['item_id_a', 'item_id_b', 'mode', 'score']]

    mega_dict = {}
    for _, row in interactions_file.iterrows():
        if row['score'] > 200:
            mega_dict.setdefault(row['item_id_a'], {}).setdefault(row['item_id_b'], []).append(row['mode'])

    return mega_dict
    # return interactions_file

def sentence_to_wordlist(sen, remove_stopwords=False):
    """Function to convert a document to a sequence of words,
    optionally removing stop words.  Returns a list of words."""
    # 2. Remove non-letters
    sentence_text = re.sub("[^a-zA-Z]", " ", sen)
    #
    # 3. Convert words to lower case and split them
    words = sentence_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    #
    # 5. Return a list of words
    return(words)

def texts_to_words(raw_text):
    """Function to convert a raw text to a string of words
       The input is a single string (a raw text), and
       the output is a single string (a preprocessed text)"""

    # 1. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", raw_text)

    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()

    # 3. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 4. Remove stop words
    meaningful_words = [w for w in words if w not in stops]
    #
    # 5. Join the words back into one string separated by space,
    # and return the result.
    return(" ".join(meaningful_words))

def produce_labels(train_dict, test_dict):
    """Produce Train and Test set labels from their corresponding dicts"""

    labels_train = []
    for value in train_dict.values():
        val = value[0]
        labels_train.append(val[-1])

    labels_test = []
    for value in test_dict.values():
        val = value[0]
        labels_test.append(val[-1])

    return labels_train, labels_test


