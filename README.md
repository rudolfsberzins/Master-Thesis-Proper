# Master-Thesis-Proper
Repository to store code for my Master Thesis

# Dependencies
    pickleshare==0.7.4 #Known as pickle
    nltk==3.2.2
    pandas==0.19.2
    numpy==1.12.0
    scipy==0.18.1
    scikit-learn==0.18.1
    xgboost==0.6a2
    gensim==1.0.1
    
    

# How to run
The code only works in Python 3.x. Running in Python 2.x is not recommened and might result in failiures 

In the folder you want to run the scripts make a folder named '\data'. In that folder you must include the following:

* A path or a link(in case of linux) to the folder containing Medline abstracts or the Medline folder itself.
* A mentions file that is the output of association tagger (link - https://bitbucket.org/larsjuhljensen/tagger)
* An X_entities.tsv file gotten from the organisms dictionary (link - https://download.jensenlab.org)
* Organisms interaction file downloaded from StringDB (link - https://www.string-db.org)

To produce the initial datasets do the following. I will use computerome_drosophila.py script from Computerome-scripts/Initiation as an example:

    #Use import sys, sys.path.insert(0, 'PATH TO CORE-SCRIPTS FOLDER')
    
    from parse_and_prepare import ProteinProteinInteractionClassifier as PPIC
    import time 
    
    def main():
        files = ['Medline',
                 'drosophila_mentions',
                 'fly_entities.tsv',
                 '7227.protein.actions.v10.txt']
        clf = PPIC(files, full_sen_set=True)
        clf.prepare()
        st = time.time() for timing
        clf.produce_pairs()
        clf.add_real_ids_and_mask()
        print('sentence_addition = %s seconds' % (time.time() - st))


    if __name__ == '__main__':
        main()
        
'full_sen_set' (default = false) designates wheter to make only the STRICT version of the data (true PPI pairs only) or make all possible models (STRICT, GENERAL (or gen) and FULL (named 'be')).

STRICT refers to a subset of the FULL dataset where only sentences with 2 protein mentions are included.
GENERAL refers to a subset of the FULL dataset where sentences with 2 or more protein mentions are included.
FULL refers to the starting mention dataset, but it is converted to a format that is easier to work with. Includes sentences with at least 1 PPI mention.

The dataset hiearchy is as follows:

FULL includes sentences from GENERAL and STRICT

GENERAL includes sentences from STRICT

STRICT is individual

**IMPORTANT - You might want to comment out line 41 and uncomment line 42 in _parse_and_perpare.py_. This is because I hardcoded the path to Medline folder in my computerome folder and it ignores the first item of the 'files' list.**

The script above will produce 6 'pickle' files. Read more about 'pickle' files here - https://docs.python.org/3.6/library/pickle.html

After the initial datasets are procuded, one can use functions in prediction.py to perform word2vec modeling and prediction with XGBoost. There are three main functions that could be of use - 

    pred.make_w2v_model(_parameters_) 
    #Make a Word2Vec model, outputs a word2vec model
    
    pred.make_models(_parameters_) 
    #Makes feature vectors from given training and testing data. 
    #Also performs make_w2v_model, but can take a pre-made word2vec model as argument, thus allowing the use of premade models,  
    #outputs a list  - [train_vecs, test_vecs, train_labels, test_labels]
    
    pred.XGB_classifier(_parameters_) 
    #Performs classification using XGBoost, outputs the error and class porbabilites

-RB

