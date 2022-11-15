from torch.utils.data import Dataset
import random
from dataclasses import dataclass

import os
from os.path import basename, isdir, isfile

from os import listdir
from os.path import isfile, join
import numpy as np

from pathlib import Path
import glob2

import torchvision.transforms as transforms

@dataclass
class Recording:
    id: int
    is_dementia: bool
    database: str
    task: str
    id_subject: int
    id_recording: int
    embed_audio_path: str
    
class MyDementiaDataset(Dataset):
    def __init__(self, data_list, transform):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        recording = self.data_list[idx]
        matrix = np.load(recording.embed_audio_path)
        matrix = self.transform(matrix)
        return (matrix, recording.is_dementia)

def get_ids_subject_and_recording(basename, database):
    subject_id, recording_id = None, None
    if not '(' in basename:
        if database == 'pitt':
            # ex format for pitt database: 001-2.npy
            subject_id_str, recording_str = basename.split('-')
            subject_id = int(subject_id_str)
            recording_id = int(recording_str)
        elif database == 'adress2020':
            # ex format for adress2020 database: S079.npy
            subject_id = int(basename.replace('S',''))
            recording_id = 0
    return subject_id, recording_id

def get_set(aDatabase, aSubjectId, split_id_subjects_per_database):
    split_dict = split_id_subjects_per_database[aDatabase]
    for aSet in ['train','valid','test']:
        if aSubjectId in split_dict[aSet]:
            return aSet

def get_all_data(aPath):

    _PATH_ = aPath
    labels = [f for f in listdir(_PATH_) if isdir(join(_PATH_, f))]

    id_trace = 0
    all_data = []
    for label in labels:
        label_path = _PATH_ + label + '/'
        databases = [f for f in listdir(label_path) if isdir(join(label_path, f))]
        for database in databases:
            database_path = label_path + database + '/'
            tasks = [f for f in listdir(database_path) if isdir(join(database_path, f))]
            for task in tasks:
                task_path = database_path + task + '/'
                filenames = [ f for f in listdir(task_path) if isfile(join(task_path, f)) ]
                for aFile in filenames:
                    id_subject, id_recording = get_ids_subject_and_recording(aFile.split('.')[0], database)
                    if id_subject != None:
                        all_data.append(
                            Recording(
                                id=id_trace,
                                is_dementia=label=='dementia',
                                database=database,
                                task=task,
                                id_subject=id_subject,
                                id_recording=id_recording,
                                embed_audio_path=task_path+aFile
                            )
                        )
                        id_trace += 1
                    else:
                        print('/!\ to remove:', aFile)
    
    return all_data


def split_dataset(aPath, transform, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, seed_random=42):
    "CROSS SUBJECT SPLIT: we make sure that id_subjects for each database don't overlap"
    
    all_data = get_all_data(aPath)

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

    train_dataset = MyDementiaDataset(data_list = training_data, transform=transform)
    valid_dataset = MyDementiaDataset(data_list = valid_data, transform=transform)
    test_dataset = MyDementiaDataset(data_list = test_data, transform=transform)

    return train_dataset, valid_dataset, test_dataset