# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.data.build_datasets import split_dataset
from src.models.cnn1_model import CNN1

from tqdm import tqdm
from sklearn import metrics

from torch import float32, nn
import torch.optim as optim

import numpy as np

def accuracy(target, pred):
    return metrics.accuracy_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())

@click.command()
@click.argument('data_path', type=click.Path())
@click.argument('models_path', type=click.Path())
@click.argument('reports_path', type=click.Path())
@click.argument('cuda', type=str)
def main(data_path, models_path, reports_path, cuda, num_epochs=10, batch_size = 32, workers = 32):
    # Define Device as cuda to use GPU
    device = torch.device(f'cuda:{cuda}')
    print('Running on device: {}'.format(device))
    
    # Load datasets
    transform = transforms.ToTensor()
    train_dataset, valid_dataset, test_dataset = split_dataset(data_path+'/', transform=transform)

    # Build dataloaders
    train_loader = DataLoader(
            dataset=train_dataset,
            num_workers=workers,
            batch_size=batch_size,
        ) 
    valid_loader = DataLoader(
            dataset=valid_dataset,
            num_workers=workers,
            batch_size=batch_size,
        ) 
    test_loader = DataLoader(
            dataset=test_dataset,
            num_workers=workers,
            batch_size=batch_size,
        )
    
    print("\nTraining data")
    print("Number of recordings:", len(train_dataset))
    print("Ratio of dementia samples:", np.mean(np.array([r.is_dementia for r in train_dataset.data_list])))
    x, y = next(iter(train_loader))
    print("Batch dimension (B x C x H x W):", x.shape)

    print("\nValidation data")
    print("Number of recordings:", len(valid_dataset))
    print("Ratio of dementia samples:", np.mean(np.array([r.is_dementia for r in valid_dataset.data_list])))
    x, y = next(iter(valid_loader))
    print("Batch dimension (B x C x H x W):", x.shape)

    print("\nTest data")
    print("Number of recordings:", len(test_dataset))
    print("Ratio of dementia samples:", np.mean(np.array([r.is_dementia for r in test_dataset.data_list])))
    x, y = next(iter(test_loader))
    print("Batch dimension (B x C x H x W):", x.shape)

    # Load Model
    model = CNN1(
        input_size = x.shape[2],
        in_channels = x.shape[3],
        out_channels1 = 2048,
        out_channels2 = 512,
        filter_size1 = 5,
        filter_size2 = 3,
        maxpool_kernel_size1 = 4,
        maxpool_kernel_size2 = 2,
        intermediate_layer_size=1000
    ).to(device)

    print(model)

    # Run a trial
    x_toy = x.to(device)
    #print('x_toy:', x_toy.shape)
    out_trial = model(x_toy)
    #print(out_trial)
    print('TRIAL: SUCCESS')

    # Training
    LEARNING_RATE = 0.00001
    REGULARIZATION = 1E-3
    #criterion =  nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    # weight_decay is equal to L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay = REGULARIZATION)

    validation_every_steps = 100
    step = 0
    cur_loss = 0

    # Initialize lists for training and validation
    train_accuracies = []
    valid_accuracies = []
    train_loss, valid_loss = [], []

    # To keep the best model
    max_valid_accuracy = -1.0
    best_model = None

    for epoch in tqdm(range(num_epochs)):
        train_accuracies_batches = []
        cur_loss = 0
        model.train()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device, dtype=float)
            # Forward pass, compute gradients, perform one training step.
            optimizer.zero_grad()
            output = model(inputs)
            output = output.view(-1)
            output = output.to(dtype=float)
            batch_loss = criterion(output, targets)
            batch_loss.backward()
            optimizer.step()
            cur_loss += batch_loss  
            # Increment step counter
            step += 1
            # Compute accuracy.
            predictions = (output > 0.5).to(dtype=int)
            train_accuracies_batches.append(accuracy(targets.to(dtype=int), predictions))
            
            if step % validation_every_steps == 0:
                # Validation
                train_accuracies.append(np.mean(train_accuracies_batches))
                train_accuracies_batches = []

                # Compute accuracies on validation set.
                valid_accuracies_batches = []
                valid_cur_loss = 0
                with torch.no_grad():
                    model.eval()
                    for inputs, targets in valid_loader:
                        inputs, targets = inputs.to(device), targets.to(device, dtype=float)
                        output = model(inputs)
                        output = output.view(-1)
                        output = output.to(dtype=float)
                        loss = criterion(output, targets)
                        valid_cur_loss += loss
                        predictions = (output > 0.5).to(dtype=int)
                        # print(targets, predictions)
                        valid_accuracies_batches.append(accuracy(targets.to(dtype=int), predictions) * len(inputs))
                        # Keep the best model
                        if (max_valid_accuracy == None) or (valid_accuracies_batches[-1] > max_valid_accuracy):
                            max_valid_accuracy = valid_accuracies_batches[-1]
                            best_model = model.state_dict()

                        model.train()
                    
                    # Append average validation accuracy to list.
                    valid_accuracies.append(np.sum(valid_accuracies_batches) / len(valid_dataset))
                    print(f"Step {step:<5}   training accuracy: {train_accuracies[-1]}")
                    print(f"             valid accuracy: {valid_accuracies[-1]}")
                
                valid_loss.append(valid_cur_loss / batch_size)
    
        train_loss.append(cur_loss / batch_size)

    print("Finished training.")

    print('train_accuracies:', train_accuracies)
    print('valid_accuracies:', valid_accuracies)

    train_loss = list(np.array([t.cpu().detach().numpy() for t in train_loss]).flatten())
    print('train_loss:', train_loss)
    
    valid_loss = list(np.array([t.cpu().detach().numpy() for t in valid_loss]).flatten())
    print('valid_loss:', valid_loss)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()