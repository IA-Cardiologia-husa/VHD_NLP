#Import libraries
import pandas as pd
import numpy as np
import os
from gensim.models import Word2Vec
import pickle

def get_w2v_features(w2v_model, sentence_group):
    """ Transform a sentence_group (containing multiple lists
    of words) into a feature vector. It averages out all the
    word vectors of the sentence_group.
    """
    words = np.concatenate(sentence_group)  # words in text
    index2word_set = set(w2v_model.wv.vocab.keys())  # words known to model
    
    featureVec = np.zeros(w2v_model.vector_size, dtype="float32")
    
    # Initialize a counter for number of words in a review
    nwords = 0
    # Loop over each word in the comment and, if it is in the model's vocabulary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            featureVec = np.add(featureVec, w2v_model[word])
            nwords += 1.

    # Divide the result by the number of words to get the average
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

def load_database():
    df = pd.read_hdf('~/projects/Conclusions_NLP_ICSV/data/data.h5', 'TRAIN')
    return df

def clean_database(df):

    return df

def process_database(df):
    
    W2V_create = fnmatch.filter(os.listdir('./intermediate/'), "W2Vmodel.pickle")
    if len(name_std) == 0:
        sentences = []
        for sentence_group in df['tokenized_sentences']:
            sentences.extend(sentence_group)


        # Initialize and train the model
        W2Vmodel = Word2Vec(sentences=sentences,
                            sg=0,
                            hs=1,
                            workers=1,
                            size=200,
                            min_count=3,
                            window=8,
                            sample=1e-3,
                            negative=5,
                            iter=6,
                            seed = 1234)

        pickle.dump(W2Vmodel, open(os.path.join("intermediate", f"W2Vmodel.pickle"), 'wb'))
    
    return df

def fillna_database(df):
    
    return df

def preprocess_filtered_database(df, wf_name):
    
    W2Vmodel = pickle.load(open(os.path.join("intermediate", f"W2Vmodel.pickle"), 'rb'))
    
    w2v_features = list(map(lambda sen_group: get_w2v_features(W2Vmodel, sen_group), df['tokenized_sentences']))
    X_w2v = np.array(list(map(np.array, w2v_features)))
    Z=pd.DataFrame(X_w2v)
    df_output = pd.concat([Z, df], axis=1)
    
    return df_output