import pickle
import logging
import file_readers as fr
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import Imputer
from nltk.tokenize import sent_tokenize
from gensim.models import word2vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

def manual_train_test_split(dataset, name_for_sets, random_state=8,
                            test_set_prop=0.3):
    """Split given dataset so no two interactions occur both in Test and in Train sets
    The code will try to get the around the proportion given by test_set_prop
    where train set will be 1-test_set_prop.

    If both_ents=True then there is only one Real ID so the full protein list
    is a set of datasets real ID's"""

    np.random.seed(random_state)

    real_x = []
    real_y = []
    for value in dataset.values():
        for val in value:
            real_x.append(val[1])
            real_y.append(val[3])
    full_protein_set = set(real_x + real_y)
    train_size = len(full_protein_set) - int(len(full_protein_set)*test_set_prop)
    train_prots = set(np.random.choice(np.array(list(full_protein_set)), train_size, replace=False))

    test_prots = full_protein_set - train_prots

    train = {}
    test = {}
    for key, value in dataset.items():
        val = value[0]
        if val[1] in train_prots and val[3] in train_prots:
            train.setdefault(key, []).append(val)
        elif val[1] in test_prots and val[3] in test_prots:
            test.setdefault(key, []).append(val)
        else:
            pass

    labels_train, labels_test = fr.produce_labels(train, test)

    pickle.dump(train, open('Results/' + name_for_sets + '_train_data.pkl', 'wb'))
    pickle.dump(test, open('Results/' + name_for_sets + '_test_data.pkl', 'wb'))
    pickle.dump(labels_train, open('Results/' + name_for_sets + '_train_labels.pkl', 'wb'))
    pickle.dump(labels_test, open('Results/' + name_for_sets + '_test_labels.pkl', 'wb'))

    return train, test, labels_train, labels_test

def make_w2v_model(dataset, name_for_model, model_features=None):
    """Produce a Word2Vec Model

    Model_features (list): Features of the word to vec models
        1. Word vector dimensionality
        2. Minimum word count
        3. Number of threads to run in parallel
        4. Context window size
        5. Downsample setting for frequent words

    """

    print ('Parsing datasets sentences')

    sentences = [fr.sentence_to_wordlist(sen) for sen in dataset]

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

    # Set values for various parameters
    if model_features:
        num_features = model_features[0] #300  # Word vector dimensionality
        min_word_count = model_features[1] #5  # Minimum word count
        num_workers = model_features[2] #4  # Number of threads to run in parallel
        context = model_features[3] #6  # Context window size
        downsampling = model_features[4] #0.001  # Downsample setting for frequent words
    else:
        num_features = 600  # Word vector dimensionality
        min_word_count = 6  # Minimum word count
        num_workers = 4  # Number of threads to run in parallel
        context = 6  # Context window size
        downsampling = 0.000001  # Downsample setting for frequent words

    print('Training Word2Vec Model')

    model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count=min_word_count, \
            window=context, sample=downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    model_name = 'Results/' + name_for_model + '_model'

    model.save(model_name)

    w2v_model = model

    return w2v_model

def bag_of_words_feat_vecs(data_train, data_test):
    """Create feature vectors using bag of words"""

    num_texts = len(data_train)

    print ('Cleaning and parsing the training set articles\n')
    clean_train_texts = []
    for idx in range(0, num_texts):
        clean_train_texts.append(fr.texts_to_words(list(data_train.keys())[idx]))

    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=300)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.

    train_data_features = vectorizer.fit_transform(clean_train_texts)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    train_data_features = train_data_features.toarray()

    num_texts = len(data_test)

    print ('Cleaning and parsing the testing set articles\n')
    clean_test_texts = []

    for idx in range(0, num_texts):
        clean_test_texts.append(fr.texts_to_words(list(data_test.keys())[idx]))

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_texts)
    test_data_features = test_data_features.toarray()

    bow_train_vecs = train_data_features
    bow_test_vecs = test_data_features

    return bow_train_vecs, bow_test_vecs

def make_feature_vec(words, model, num_features):
    """Function to average all of the word vectors in a given
    paragraph"""

    #
    # Pre-initialize an empty numpy array (for speed)
    feature_vec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            feature_vec = np.add(feature_vec,model[word])
    #
    # Divide the result by the number of words to get the average
    feature_vec = np.divide(feature_vec,nwords)
    return feature_vec


def get_avg_feature_vecs(sentences, model, num_features):
    """Given a set of reviews (each one a list of words), calculate
    the average feature vector for each one and return a 2D numpy array"""
    #
    # Initialize a counter
    counter = 0
    #
    # Preallocate a 2D numpy array, for speed
    sentence_feature_vecs = np.zeros((len(sentences), num_features), dtype="float32")
    #
    # Loop through the reviews
    for sen in sentences:
        # Call the function (defined above) that makes average feature vectors
        sentence_feature_vecs[counter] = make_feature_vec(sen, model, \
           num_features)

        # Increment the counter
        counter = counter + 1
    return sentence_feature_vecs


def word_2_vec_feat_vecs(data_train, data_test, model, feature_count=300):
    """Produce Word 2 Vec Feature vectors"""

    clean_train_texts = [fr.sentence_to_wordlist(sen, remove_stopwords=True) for sen in data_train]

    w2v_train_vecs = get_avg_feature_vecs(clean_train_texts, model, feature_count)

    # w2v_train_df = pd.DataFrame(data=w2v_train_vecs)

    clean_test_texts = [fr.sentence_to_wordlist(sen, remove_stopwords=True) for sen in data_test]

    w2v_test_vecs = get_avg_feature_vecs(clean_test_texts, model, feature_count)

    # w2v_test_df = pd.DataFrame(data=w2v_test_vecs)

    imputer = Imputer()

    w2v_train_transf = imputer.fit_transform(w2v_train_vecs)
    w2v_test_transf = imputer.fit_transform(w2v_test_vecs)

    return w2v_train_transf, w2v_test_transf


def XGB_classifier(train_vector, test_vector,
                   labels_train, labels_test,
                   feature_selection=False):
    """Perform XGB Classification"""
    if feature_selection:
        clf = ExtraTreesClassifier(n_estimators=100)
        clf = clf.fit(train_vector, labels_train)
        model = SelectFromModel(clf, prefit=True)
        train_vector = model.transform(train_vector)
        test_vector = model.transform(test_vector)

    xgb_clf = xgb.XGBClassifier()
    print ("\n Fitting XGBoost Model!")
    xgb_clf = xgb_clf.fit(train_vector, labels_train)
    print ("\n Making Predictions")
    result = xgb_clf.predict(test_vector)
    probs = xgb_clf.predict_proba(test_vector)[:, 1]
    predictions = [round(val) for val in result]
    error = get_accuracy(predictions, labels_test)

    return error, probs

def get_accuracy(l_new, l_te):
    """Calculates the accuracy of predicted labels, based on the given labels

    INPUT: New(Predicted) Labels, Test Labels

    OUTPUT: Error  """

    acc = 0

    for i in range(len(l_te)):
        if l_new[i] == l_te[i]:
            acc += 1

    acc = float(acc / len(l_te))

    return 1-acc



def make_models(init_dataset, dataset_name,
                prev_model=None, ran_state=8, BOW=False):
    """Combines all function in a single call

    prev model used when using Both Entities dict

    Returns a list of feature vectors and labels. Order:
    0 - bow_train_vecs
    1 - bow_test_vecs
    2 - w2v_train_vecs
    3 - w2v_test_vecs
    4 - labels_train
    5 - labels_test"""
    if prev_model is not None:
        w2v_model = prev_model
    else:
        w2v_model = make_w2v_model(init_dataset, dataset_name)

    train_data, test_data, labels_train, labels_test = manual_train_test_split(init_dataset, dataset_name, random_state=ran_state)

    if BOW:
        bow_train_vecs, bow_test_vecs = bag_of_words_feat_vecs(train_data, test_data)
        w2v_train_vecs, w2v_test_vecs = word_2_vec_feat_vecs(train_data, test_data, w2v_model)

        result_list = [bow_train_vecs, bow_test_vecs,
                       w2v_train_vecs, w2v_test_vecs,
                       labels_train, labels_test]

        pickle.dump(result_list, open('Results/' + dataset_name + 'w_BOW_results_list.pkl', 'wb'))
    else:
        w2v_train_vecs, w2v_test_vecs = word_2_vec_feat_vecs(train_data, test_data, w2v_model)

        result_list = [w2v_train_vecs, w2v_test_vecs,
                       labels_train, labels_test]

        pickle.dump(result_list, open('Results/' + dataset_name + 'no_BOW_results_list.pkl', 'wb'))

    return result_list



