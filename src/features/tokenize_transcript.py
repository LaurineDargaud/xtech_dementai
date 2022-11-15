# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import glob2
import numpy as np

from os import makedirs
from os.path import isdir

import torch
from transformers import DistilBertTokenizer, DistilBertModel

from tqdm import tqdm

import pickle

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('cuda', type=str)
def main(input_filepath, output_filepath, cuda, tokenizer_label='distilbert-base-uncased'):
    """ Turn transcripts from (../interim) into
        tokens for classification (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('tokenization of transcripts')

    # get all files in input_filepaths
    all_files = [ Path(p).absolute() for p in glob2.glob(input_filepath + '/*') ]
    print('Number of files:', len(all_files))

    # create target output folder
    if not isdir(output_filepath):
        makedirs(output_filepath)

    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Load the BERT tokenizer
    print('Loading Bert tokenizer...')
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_label)
    model = DistilBertModel.from_pretrained(tokenizer_label)

    # transform and save each transcript
    for aTranscriptPath in tqdm(all_files):
        # get transcript txt
        with open(aTranscriptPath, 'rb') as handle:
            pickle_file = pickle.load(handle)
        
        # add [CLS] at the beginning and [SEP] of the end of each sentence
        final_transcript = '[CLS]' + ' [SEP]'.join([a['text'] for a in pickle_file['segments']]) + ' [SEP]'
        
        # tokenize with max_length=500
        tokens = tokenizer(final_transcript)
        inputs = tokenizer(final_transcript, return_tensors="pt", truncation=True, max_length=500, padding='max_length')
        
        # apply DistilBert model
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        
        # save last hidden states
        with open(output_filepath+'/'+aTranscriptPath.stem+'.npy', 'wb') as f:
            np.save(f, last_hidden_states.cpu().detach().numpy())        

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
