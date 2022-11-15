from dataclasses import dataclass

@dataclass
class Transcript:
    id: int
    is_dementia: int
    database: str
    task: str
    id_subject: int
    id_recording: int
    embedding_vector_path: str
    mmse_score: int

from torch.utils.data import Dataset
import pickle

class MyDementiaTranscriptDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        transcript = self.data_list[idx]
        matrix = np.load(transcript.embedding_vector_path)
        return (matrix, transcript.is_dementia)

from src.data.build_datasets import get_ids_subject_and_recording, get_set

from os.path import isdir, isfile, join
from os import listdir

import numpy as np

import random
import pandas as pd

def get_mmse_column_name(anId):
    return 'mmse'+str(anId+1)

def mmse_label_from_score(aScore):
    """
    if aScore >= 24:
        return 0
    if aScore >= 19:
        return 1
    if aScore >= 10:
        return 2
    return 3
    """
    return int(aScore < 19)

def get_all_data(aPath, givenDatabase=None, givenTask=None):

    _PATH_ = aPath
    labels = [f for f in listdir(_PATH_) if isdir(join(_PATH_, f))]
    
    summary = pd.read_csv(aPath+'../Pitt-data-summary.csv', delimiter=';')
    summary = summary.drop('Unnamed: 16', axis=1)

    id_trace = 0
    all_data = []
    for label in labels:
        label_path = _PATH_ + label + '/'
        if givenDatabase != None:
            databases = [givenDatabase]
        else:
            databases = [f for f in listdir(label_path) if isdir(join(label_path, f))]
        for database in databases:
            database_path = label_path + database + '/'
            if givenDatabase != None:
                tasks = [givenTask]
            else:
                tasks = [f for f in listdir(database_path) if isdir(join(database_path, f))]
            for task in tasks:
                task_path = database_path + task + '/'
                filenames = [ f for f in listdir(task_path) if isfile(join(task_path, f)) ]
                for aFile in filenames:
                    id_subject, id_recording = get_ids_subject_and_recording(aFile.split('.')[0], database)
                    if id_subject != None:
                        # change is_dementia to mmse score provided in Pitt-summary-data
                        column_to_track = get_mmse_column_name(id_recording)
                        mmse_score = summary[summary.id == id_subject][column_to_track].values
                        isDementia = summary[summary.id == id_subject]['isDementia'].values
                        if len(mmse_score) == 1:
                            mmse_score = mmse_score[0]
                            all_data.append(
                                Transcript(
                                    id=id_trace,
                                    is_dementia= int(label == 'dementia'), #int(isDementia[0]!=0) #mmse_label_from_score(mmse_score)
                                    database=database,
                                    task=task,
                                    id_subject=id_subject,
                                    id_recording=id_recording,
                                    embedding_vector_path=task_path+aFile,
                                    mmse_score = mmse_score
                                )
                            )
                            id_trace += 1
                        else:
                            print('ERROR to get MMSE:', aFile)
                    else:
                        print('/!\ to remove:', aFile)
    
    return all_data, databases

def split_dataset(aPath, train_ratio=0.8, valid_ratio=0.0, test_ratio=0.2, seed_random=42):
    "CROSS SUBJECT SPLIT: we make sure that id_subjects for each database don't overlap"
    
    all_data, databases = get_all_data(aPath)

    #building training, validation and test sets
    ratios = {
        'train':train_ratio,
        'valid':valid_ratio,
        'test':test_ratio
    }
    assert sum(list(ratios.values())) == 1.0

    all_id_subject_per_database = {}

    for database in databases:
        ids = [r.id_subject for r in all_data if r.database == database]
        all_id_subject_per_database[database]=list(set(ids))
        # print('Nb subjects for', database,':',len(list(set(ids))))
    
    split_id_subjects_per_database = {}

    for k, aList in all_id_subject_per_database.items():
        random.seed(seed_random)
        random.shuffle(aList)
        total_size = len(aList)
        index_training = int(total_size*ratios['train'])
        index_valid = index_training + int(total_size*ratios['valid'])
        idx_training = aList[:index_training]
        idx_valid = aList[index_training:index_valid]
        idx_test = aList[index_valid:]
        split_dict = {
            'train':idx_training,
            'valid':idx_valid,
            'test':idx_test
        }
        split_id_subjects_per_database[k] = split_dict

    assert len(np.intersect1d(split_id_subjects_per_database['pitt']['train'], split_id_subjects_per_database['pitt']['test'])) == 0

    training_data, valid_data, test_data = [], [], []

    for aRecording in all_data:
        defined_set = get_set(aRecording.database, aRecording.id_subject, split_id_subjects_per_database)
        if defined_set == 'train':
            training_data.append(aRecording)
        elif defined_set == 'valid':
            valid_data.append(aRecording)
        else:
            test_data.append(aRecording)

    # print('SIZES:')
    # print('Training data:', len(training_data))
    # print('Validation data:', len(valid_data))
    # print('Test data:', len(test_data))

    train_dataset = MyDementiaTranscriptDataset(data_list = training_data)
    valid_dataset = MyDementiaTranscriptDataset(data_list = valid_data)
    test_dataset = MyDementiaTranscriptDataset(data_list = test_data)

    return train_dataset, valid_dataset, test_dataset