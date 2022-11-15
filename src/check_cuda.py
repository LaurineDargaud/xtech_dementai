from joblib import effective_n_jobs

CUDA_VISIBLE_DEVICES=0,1

print('Nb of effective jobs: ', effective_n_jobs(-1), 'jobs')

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

print('Nb devices:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))