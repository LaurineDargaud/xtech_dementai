# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from joblib import load
import numpy as np

from os import makedirs
from os.path import isdir

import whisper
import torch

from transformers import DistilBertTokenizer, DistilBertModel

import pickle

from src.data.get_trancript_dataloader import get_all_data, MyDementiaTranscriptDataset

@click.command()
@click.argument('audio_input_file', type=click.Path(exists=True))
@click.argument('output_folder_path', type=click.Path())
@click.argument('models_path', type=click.Path())
@click.argument('cuda', type=str)
def main(audio_input_file, output_folder_path, models_path, cuda):
    """ From audio to probability
    """
    
    audio_input_file = Path(audio_input_file).absolute()
    print('\nAudio File Input File:', str(audio_input_file),'\n')
    
    tokenizer_label='distilbert-base-uncased'
    model_name = 'gp_distilBert'

    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')
    
    # create target output folder
    if not isdir(output_folder_path):
        makedirs(output_folder_path)

    # get Whisper model
    print('1. Whisper AI transcription')
    print('1.a. Loading Whisper AI...')
    model = whisper.load_model("medium.en").to(device)
    
    # transcription
    print('1.b. Run transcription...')
    result = model.transcribe(str(audio_input_file))
    # savetranscript
    with open(output_folder_path+'/'+audio_input_file.stem+'_whisperAI_transcript.pickle', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print('\nTRANSCRIPT:', result['text'],'\n')
    
    # Load the BERT tokenizer
    print('2. DistilBert tokenization')
    print('2.a. Load DistilBert...')
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_label)
    model = DistilBertModel.from_pretrained(tokenizer_label)
    
    # add [CLS] at the beginning and [SEP] of the end of each sentence
    final_transcript = '[CLS]' + ' [SEP]'.join([a['text'] for a in result['segments']]) + ' [SEP]'
    
    # tokenize with max_length=500
    print('2.a. Run tokenization...')
    inputs = tokenizer(final_transcript, return_tensors="pt", truncation=True, max_length=500, padding='max_length')
    
    # apply DistilBert model
    print('2.b. Get DistilBert embeddings...')
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    last_hidden_states = last_hidden_states.cpu().detach().numpy()
    
    # save last hidden states
    with open(output_folder_path+'/'+audio_input_file.stem+'_BertTokens.npy', 'wb') as f:
        np.save(f, last_hidden_states)  
    
    # Build X
    mean_embed = np.mean(last_hidden_states[0], axis=0)
    X = [mean_embed]

    # Load model
    print('3. Gaussian Process Classifier prediction')
    model = load(models_path+'/'+model_name+'.joblib')

    # Predict
    y_hat = model.predict_proba(X)
    print('\nProbability to have dementia:', y_hat[0][1])
    
    with open(output_folder_path+'/'+audio_input_file.stem+'_prediction.npy', 'wb') as f:
        np.save(f, y_hat)  
        
    print('OVER.')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
