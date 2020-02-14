# In this archive we have to define the dictionary ml_info. This is a dictionary of dictionaries, that for each of the ML models we want
# assigns a dictionary that contains:
#
# clf: a scikit-learn classifier, or any object that implements the functions fit, and predict_proba or decision_function in the same way.
# formal_name: name to be used in plots and report
#
# In this archive we provide 4 examples:
# RF for Random Forest
# BT for Boosted Trees
# LR for Logistic Regression
# RF_pipeline for a Random Forest with hyperparameter tuning including the choice of feature selection strategy

import sklearn.ensemble as sk_en
import sklearn.linear_model as sk_lm
import sklearn.model_selection as sk_ms
import sklearn.svm as sk_svm
import sklearn.naive_bayes as sk_nb
import sklearn.pipeline as sk_pl
import sklearn.preprocessing as sk_pp 
import numpy as np
import pandas as pd
# import xgboost as xgb
# from utils.featureselecter import FeatureSelecter
from gensim.models import Word2Vec



ML_info ={}

pipeline_svm = sk_pl.Pipeline(steps=[("slm",sk_pp.StandardScaler()),("svm",sk_svm.SVC())])

pipeline_rf = sk_pl.Pipeline(steps=[("rf",sk_en.RandomForestClassifier())])

pipeline_nb = sk_pl.Pipeline(steps=[("nb",sk_nb.GaussianNB())])

pipeline_lr = sk_pl.Pipeline(steps=[("lr",sk_lm.LogisticRegression())])


grid_params_svm=[{'svm__C':[1,10,50],
                  'svm__kernel':['linear'],
                  'svm__class_weight':['balanced'],
                  'svm__probability':[True]}]

grid_params_rf=[{'rf__n_estimators':[100,200],
                 'rf__max_features':['auto'],
                 'rf__criterion':['entropy'],
                 'rf__max_depth':[2, None]}]

grid_params_nb=[{}]

grid_params_lr=[{'lr__penalty':['l1'],
                 'lr__solver':['saga'],
                 'lr__class_weight':['balanced']}]


tuned_svm=sk_ms.GridSearchCV(pipeline_svm,grid_params_svm, cv=10,scoring ='roc_auc', return_train_score=False, verbose=1)

tuned_rf=sk_ms.GridSearchCV(pipeline_rf,grid_params_rf, cv=10,scoring ='roc_auc', return_train_score=False, verbose=1)

tuned_nb=sk_ms.GridSearchCV(pipeline_nb,grid_params_nb, cv=10,scoring ='roc_auc', return_train_score=False, verbose=1)

tuned_lr=sk_ms.GridSearchCV(pipeline_lr,grid_params_lr, cv=10,scoring ='roc_auc', return_train_score=False, verbose=1)


ML_info['SVM_pipeline'] = {'formal_name': 'Support Vector Machine (Hyperparameter Tuning)',
                          'clf': tuned_svm}

ML_info['RF_pipeline'] = {'formal_name': 'Random Forest (Hyperparameter Tuning)',
                          'clf': tuned_rf}

ML_info['NB_pipeline'] = {'formal_name': 'Naive Bayes (Hyperparameter Tuning)',
                         'clf': tuned_nb}

ML_info['LR_pipeline'] = {'formal_name': 'Logistic Regresion (Hyperparameter Tuning)',
                          'clf': tuned_lr}

