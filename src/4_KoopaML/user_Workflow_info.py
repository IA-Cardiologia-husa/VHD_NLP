# formal_title:Title for plots and reports
# label_name:Variable to predict in this workflow, e.g.: 'Var17'
# feature_list:List of features to use in the ML models, e.g.: ['Var1', 'Var2', 'Var4']
# filter_function:Function to filter the Dataframe. If we want to use only the subject of the dataframe with Var3=1, we would write: lambda df: df.loc[df['Var3']==1].reset_index(drop=True) In case we want no filter, we have to write: lambda df: df
# group_label:groups for cross-validation. Subjects from the same groups will appear in the same folds
# validation_type: "kfold", "groupkfold", "stratifiedkfold", "stratifiedgroupkfold", "unfilterdkfold" (for doing the kfold first and then filtering the folds)
# cv_folds:For kfolds, the number of folds
# cv_repetitions:For kfolds, the number of repetitions
# external_validation: 'Yes' or 'No', in case of 'Yes', you have to fill user_external_data_utils.py

WF_info = {}

WF_info['insf_Mv'] = {'formal_title': 'Insuficiencia Mitral',
                        'label_name': 'insf_Mv',
                        'feature_list': range(200),
                        'filter_function': lambda df: df.loc[df['insf_Mv'].notnull(),],
                        'group_label': None,
                        'validation_type':'stratifiedkfold',
                        'cv_folds': 10,
                        'cv_repetitions': 10,
                        'external_validation': 'Yes'}

WF_info['insf_Ao'] = {'formal_title': 'Insuficiencia Aortica',
                        'label_name': 'insf_Ao',
                        'feature_list': range(200),
                        'filter_function': lambda df: df.loc[df['insf_Ao'].notnull(),],
                        'group_label': None,
                        'validation_type':'stratifiedkfold',
                        'cv_folds': 10,
                        'cv_repetitions': 10,
                        'external_validation': 'Yes'}

WF_info['est_Mv'] = {'formal_title': 'Estenosis Mitral',
                        'label_name': 'est_Mv',
                        'feature_list': range(200),
                        'filter_function': lambda df: df.loc[df['est_Mv'].notnull(),],
                        'group_label': None,
                        'validation_type':'stratifiedkfold',
                        'cv_folds': 10,
                        'cv_repetitions': 10,
                        'external_validation': 'Yes'}

WF_info['est_Ao'] = {'formal_title': 'Estenosis Aortica',
                        'label_name': 'est_Ao',
                        'feature_list': range(200),
                        'filter_function': lambda df: df.loc[df['est_Ao'].notnull(),],
                        'group_label': None,
                        'validation_type':'stratifiedkfold',
                        'cv_folds': 10,
                        'cv_repetitions': 10,
                        'external_validation': 'Yes'}

WF_info['prot_Mv'] = {'formal_title': 'Valvula Mitral modificada',
                        'label_name': 'prot_Mv',
                        'feature_list': range(200),
                        'filter_function': lambda df: df.loc[df['prot_Mv'].notnull(),],
                        'group_label': None,
                        'validation_type':'stratifiedkfold',
                        'cv_folds': 10,
                        'cv_repetitions': 10,
                        'external_validation': 'Yes'}

WF_info['prot_Ao'] = {'formal_title': 'Valvula Aortica modificada',
                        'label_name': 'prot_Ao',
                        'feature_list': range(200),
                        'filter_function': lambda df: df.loc[df['prot_Ao'].notnull(),],
                        'group_label': None,
                        'validation_type':'stratifiedkfold',
                        'cv_folds': 10,
                        'cv_repetitions': 10,
                        'external_validation': 'Yes'}