# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.data.get_trancript_dataloader import split_dataset

from sklearn.model_selection import GridSearchCV

from joblib import dump

import pandas as pd

# we only focus on Pitt/Cookie for now

def get_classifier(aKeyWord):
    parameters = None
    if aKeyWord == 'lr':
        from sklearn.linear_model import LogisticRegression
        parameters = {'solver':('lbfgs', 'liblinear', 'saga', 'sag', 'newton-cg'), 'C':[1, 10]}
        m = LogisticRegression()
    if aKeyWord == 'svc':
        from sklearn.svm import SVC
        m = SVC()
    if aKeyWord == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        """parameters = { 
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth' : [4,5,6,7,8],
            'criterion' :['gini', 'entropy']
        }"""
        m = RandomForestClassifier()
    if aKeyWord == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        m = KNeighborsClassifier()
    if aKeyWord == 'mlp':
        from sklearn.neural_network import MLPClassifier
        m = MLPClassifier()
    if aKeyWord == 'gp':
        from sklearn.gaussian_process import GaussianProcessClassifier
        m = GaussianProcessClassifier()
    if aKeyWord == 'dt':
        from sklearn.tree import DecisionTreeClassifier
        m = DecisionTreeClassifier()
    return m, parameters


@click.command()
@click.argument('data_path', type=click.Path())
@click.argument('models_path', type=click.Path())
@click.argument('reports_path', type=click.Path())
@click.argument('classifier_type', type=str)
def main(data_path, models_path, reports_path, classifier_type = 'lr'):
    
    # Load datasets
    train_dataset, _, test_dataset = split_dataset(data_path+'/')
    train_list, test_list = train_dataset.data_list, test_dataset.data_list

    audio_features_type = 'Functionals'
    dementia_features = pd.read_csv(data_path+f'/../pitt_cookie_dementia_opensmile_ComParE_2016_{audio_features_type}_features.csv')
    control_features = pd.read_csv(data_path+f'/../pitt_cookie_control_opensmile_ComParE_2016_{audio_features_type}_features.csv')

    dementia_features['hasDementia'] = 1
    control_features['hasDementia'] = 0

    df = pd.concat([dementia_features,control_features])

    import pdb; pdb.set_trace();

    # Build Xtrain and Xtest for Logisitc Regression

    def get_features_from_item(anItem):
        hasDementia, vector_path = anItem.is_dementia, anItem.embedding_vector_path
        filename = vector_path.split('/')[-1].split('.')[0] + '.mp3'
        df_row = df[(df.fileName == filename) & (df.hasDementia == int(hasDementia))]
        df_row = df_row.drop(['fileName','hasDementia','Unnamed: 0'], axis=1)
        assert len(df_row)==1
        return df_row.iloc[0].values, hasDementia

    X_train, y_train = [], []
    for anItem in train_list:
        aMatrix, aLabel = get_features_from_item(anItem)
        X_train.append(aMatrix)
        y_train.append(aLabel)
    
    X_test, y_test = [], []
    for anItem in test_list:
        aMatrix, aLabel = get_features_from_item(anItem)
        X_test.append(aMatrix)
        y_test.append(aLabel)
    
    print('Shape X_train:', len(X_train), 'x', len(X_train[0]))
    print('Shape X_test:', len(X_test), 'x', len(X_test[0]))

    # Train LR
    model, parameters = get_classifier(classifier_type)
    print('Classifier:', classifier_type)

    if parameters != None:
        clf = GridSearchCV(model, parameters)
        print('GridsearchCV in progress...')
        clf = clf.fit(X_train, y_train)
        model = clf.best_estimator_
        print('Best Params', clf.best_params_)
        classifier_type +='_best'
    else:
        model = model.fit(X_train, y_train)

    print('Accuracy on training set:', model.score(X_train, y_train))

    # Predict LR
    print('Accuracy on test set:', model.score(X_test, y_test))

    # Save Best Model
    dump(model, models_path+f'/{classifier_type}_audio_{audio_features_type}.joblib')

    # import pdb; pdb.set_trace();


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()