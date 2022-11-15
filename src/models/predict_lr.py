# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.data.get_trancript_dataloader import get_all_data, MyDementiaTranscriptDataset
from src.data.get_trancript_dataloader import split_dataset

from joblib import load

import numpy as np

@click.command()
@click.argument('data_path', type=click.Path())
@click.argument('models_path', type=click.Path())
@click.argument('reports_path', type=click.Path())
@click.argument('model_name', type=str)
@click.argument('database', type=str)
@click.argument('task', type=str)
def main(data_path, models_path, reports_path, model_name, database, task):
    
    # Load dataset
    if database == 'pitt' and task=='cookie':
        _, _, dataset = split_dataset('data/processed/')
    else:
        all_data, _ = get_all_data(data_path+'/', givenDatabase=database, givenTask=task)
        dataset = MyDementiaTranscriptDataset(data_list = all_data)

    # Build X and y for Logisitc Regression

    X, y = [], []
    for _, anItem in enumerate(dataset):
        aMatrix, aLabel = anItem
        aMatrix = aMatrix[0]
        mean_embed = np.mean(aMatrix, axis=0)
        X.append(mean_embed)
        y.append(aLabel)
    
    print('Shape dataset:', len(X), 'x', len(X[0]))

    # Load model
    model = load(models_path+'/'+model_name+'.joblib')

    # Predict
    print('Accuracy on test set:', model.score(X, y))
    
    # Save probas
    y_probs = model.predict_proba(X)
    np.save(reports_path+f'/y_probs_{database}_{task}.npy', y_probs)
    np.save(reports_path+f'/y_true_{database}_{task}.npy', y)
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()