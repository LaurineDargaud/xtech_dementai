# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from sklearn.linear_model import LogisticRegression

from src.data.get_trancript_dataloader import split_dataset

from sklearn.model_selection import GridSearchCV

from tqdm import tqdm

from joblib import dump

import numpy as np

def get_classifier(aKeyWord):
    parameters = None
    if aKeyWord == 'lr':
        from sklearn.linear_model import LogisticRegression
        parameters = {'solver':('lbfgs', 'liblinear', 'saga', 'sag', 'newton-cg'), 'C':[1, 10]}
        m = LogisticRegression(max_iter=10000)
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
def main(data_path, models_path, reports_path, classifier_type = 'gp'):
    
    # Load datasets
    train_dataset, _, test_dataset = split_dataset(data_path+'/')

    # Build Xtrain and Xtest for Logisitc Regression

    X_train, y_train = [], []
    for i, anItem in enumerate(train_dataset):
        aMatrix, aLabel = anItem
        aMatrix = aMatrix[0]
        mean_embed = np.mean(aMatrix, axis=0)
        X_train.append(mean_embed)
        y_train.append(aLabel)
    
    X_test, y_test = [], []
    for i, anItem in enumerate(test_dataset):
        aMatrix, aLabel = anItem
        aMatrix = aMatrix[0]
        mean_embed = np.mean(aMatrix, axis=0)
        X_test.append(mean_embed)
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
    dump(model, models_path+f'/{classifier_type}_distilBert.joblib')

    # Save probas
    test_probas = model.predict_proba(X_test)
    np.save(models_path+f'/{classifier_type}_distilBert_test_probas.npy', test_probas)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()