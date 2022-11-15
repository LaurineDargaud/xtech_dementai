# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import glob2
import numpy as np

from os import makedirs
from os.path import isdir

import whisper
import torch

from tqdm import tqdm

import pickle

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('cuda', type=str)
def main(input_filepath, output_filepath, cuda):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('transcription of recordings')

    # get all files in input_filepaths
    all_files = [ Path(p).absolute() for p in glob2.glob(input_filepath + '/*') ]
    print('Number of files:', len(all_files))

    # create target output folder
    if not isdir(output_filepath):
        makedirs(output_filepath)

    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # get Whisper model
    print('Loading Whisper model...')
    model = whisper.load_model("medium.en").to(device)

    # transform and save each file
    for audioFile in tqdm(all_files):
        # transcription
        result = model.transcribe(str(audioFile))
        # save embed audio
        target_path = output_filepath+'/'+audioFile.stem+'.pickle'
        with open(target_path, 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
