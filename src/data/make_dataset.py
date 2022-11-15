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

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('cuda', type=str)
def main(input_filepath, output_filepath, cuda):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

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
        # print('Processing:', audioFile,'...')
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(audioFile)
        audio = whisper.pad_or_trim(audio)
        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        mel = mel.view(1, mel.shape[0], mel.shape[1])
        # get embed audio
        transformed_audio = model.embed_audio(mel)
        transformed_audio = transformed_audio.cpu().detach().numpy()[0]
        # save embed audio
        with open(output_filepath+'/'+audioFile.stem+'.npy', 'wb') as f:
            np.save(f, transformed_audio)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
