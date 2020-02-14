#Import libraries
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import os
import pickle

tmp_path = os.path.abspath("intermediate")

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

def load_external_database():
    df = pd.read_hdf('~/projects/Conclusions_NLP_ICSV/data/test.h5', 'TEST')
    return df

def clean_external_database(df):
    return df

def process_external_database(df):
    return df

def fillna_external_database(df):
    return df

def preprocess_filtered_external_database(df, wf_name):
    W2Vmodel = pickle.load(open(os.path.join("intermediate", f"W2Vmodel.pickle"), 'rb'))
    if(wf_name == 'prot_Ao'):
        df = df.loc[df['prot_Ao'].notnull(),]
    elif(wf_name == 'prot_Mv'):
        df = df.loc[df['prot_Mv'].notnull(),]
    elif(wf_name == 'insf_Ao'):
        df = df.loc[df['insf_Ao'].notnull(),]
    elif(wf_name == 'insf_Mv'):
        df = df.loc[df['insf_Mv'].notnull(),]
    elif(wf_name == 'est_Ao'):
        df = df.loc[df['est_Ao'].notnull(),]
    elif(wf_name == 'est_Mv'):
        df = df.loc[df['est_Mv'].notnull(),] 
    else:
        raise('No se ha ejecutado los if')
    df = df.reset_index(drop=True)

    w2v_features = list(map(lambda sen_group: get_w2v_features(W2Vmodel, sen_group), df['tokenized_sentences']))
    X_w2v = np.array(list(map(np.array, w2v_features)))
    Z=pd.DataFrame(X_w2v)
    df_output = pd.concat([Z, df], axis=1)
    
    return df_output
