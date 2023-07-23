"""
The script trains a specified deep learning model for a binary classification problem, using a dataset stored in HDF5 format. The deep learning model can be one of 
several types, including GoogleNet, ResNet, SqueezeNet, DenseNet, VGG, AlexNet, and Inception3. The script allows for various parameters to be specified, such as 
the number of training epochs, batch size, learning rate, optimizer, and whether to use a scheduler, data augmentation, and weight decay. The script also includes 
support for training on GPUs and for loading a previously trained model for fine-tuning or resuming training.

This script is run by `bash scripts/train.sh`

"""
import os
import argparse
import numpy as np
from pathlib import Path
from termcolor import colored

import torch
import torchvision.transforms as tvtrans
import tensorboard

from utils import (getTimestamp, getHostName, makeSubdir,
                   logInfoWithDot, printArgs, set_logging)
from dataloader import HyphalDataset
from solver import HyphalSolver

import optuna
import pandas as pd

parser = argparse.ArgumentParser()
# Model
parser.add_argument('--model_type', default='GoogleNet', help='model used for training', choices=['GoogleNet', 'ResNet', 'SqueezeNet', 'DenseNet', 'VGG', 'AlexNet', 'Inception3'])
parser.add_argument('--pretrained', action='store_true', help='use pretrained model parameters')
parser.add_argument('--feature_extract', action='store_true', help='fine-tune the last layer only')
parser.add_argument('--resume', action='store_true', help='resume training') 
parser.add_argument('--resume_timestamp', help='timestamp to resume')
parser.add_argument('--loading_epoch', type=int, default=0, help='xth model loaded to resume')
parser.add_argument('--total_epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--outdim', type=int, default=2, help='number of classes')
parser.add_argument('--save_model', action='store_true', help='save model')
parser.add_argument('--cuda', action='store_true', help='enable cuda')
parser.add_argument('--mps', action='store_true', help='enable mps')
parser.add_argument('--means', type=float, nargs='+', default="0.504 0.604 0.361",
                    help='List of means for each channel')
parser.add_argument('--stds', type=float, nargs='+', default="0.144 0.142 0.192",
                    help='List of standard deviations for each channel')


# Optimizer
parser.add_argument('--optim_type', default='Adam', help='optimizer used for training', choices=['Adam', 'Adadelta', 'RMSprop', 'SGD'])
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for optimzer')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay for optimizer')
parser.add_argument('--weighted_loss', action='store_true', help='weighted loss')

# Scheduler
parser.add_argument('--scheduler', action='store_true', help='use scheduler')
parser.add_argument('--step_size', type=int, default=60, help='period of learning rate decay')
parser.add_argument('--gamma', type=float, default=0.5, help='multiplicative factor of learning rate decay')

# Dataloader
parser.add_argument('--bsize', type=int, default=32, help='input batch size')
parser.add_argument('--nworker', type=int, default=2, help='number of dataloader workers')
parser.add_argument('--manual_seed', type=int, default=1701, help='reproduce experiemnt')
parser.add_argument('--cuda_device', default="0", help='ith cuda used for training')
parser.add_argument('--root_path', type=str, required=True, help='path to data')
parser.add_argument('--test_date', type=str, help='date to be tested')
parser.add_argument('--qtl_partition_idx', type=str, help='qtl partition to be used')
parser.add_argument('--seg_idx', type=str, help='segmentation index to be used in the cross-validation of the deep learning training script')
parser.add_argument('--demo_dataset', action='store_true', help='use balanced dataset')
parser.add_argument('--seg_dataset', action='store_true', help='use randomized dataset')
parser.add_argument('--aug_dataset', action='store_true', help='use augmented dataset')
parser.add_argument('--cross_validation', action='store_true', help='use cross validation dataset')

# Fortuna
parser.add_argument('--n_trials', type=int, default=25, help='trials for fortuna')
parser.add_argument('--study_name', type=str, default='DownyTrial', help='name of study for fortuna')


opt = parser.parse_args()

np.random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)
torch.cuda.manual_seed(opt.manual_seed)

if opt.mps:
    device_type = "mps"
elif opt.cuda:
    device_type = "cuda"
else:
    device_type = "cpu"

gpu = torch.device(device_type)

# Basic configuration
root_path = Path(opt.root_path)
result_root_path = root_path / 'results'
dataset_root_path = root_path / 'data'

if opt.demo_dataset:
    meta_filepath = 'metadata.csv'
    train_filepath = 'demo_train_set.hdf5'
    test_filepath = 'demo_val_set.hdf5'
    group = 'demo'
elif opt.aug_dataset:
    meta_filepath = 'metadata.csv'
    train_filepath = 'train_set_aug.hdf5'
    test_filepath = 'test_set_aug.hdf5'
else:
    meta_filepath = 'metadata.csv'
    train_filepath = 'train.hdf5'
    test_filepath = 'test.hdf5'
    group = 'asabe_journal'

if opt.cross_validation:
    dataset_root_path = dataset_root_path / 'cross_validation_ds' / opt.test_date
    group = 'cls_cv'
elif opt.qtl_partition_idx:
    dataset_root_path = dataset_root_path / \
        'qtl_partition_test' / f'partition_ratio_{opt.qtl_partition_idx}'
    group = 'qtl_partition'
elif opt.seg_dataset:
    dataset_root_path = dataset_root_path
    group = 'cls_seg_dataset'

#print("Dataset path:", dataset_root_path)

# Dataloader
bsize = opt.bsize
nworker = opt.nworker

# Model file format and location configuration
year = '2023' # this is for labeling the model
current_time = getTimestamp() if not opt.resume else opt.resume_timestamp
model_type_time = opt.model_type + '_{0}_{1}'.format(current_time, year)
model_path = result_root_path / 'models' / model_type_time
model_filename = '{0}_model_ep{1:03}'

# Logger and Writer
log_path = result_root_path / 'logs' / group / model_type_time
log_filename = 'log_{0}_{1}_{2}.txt'.format(
    'train', current_time, year)  # log_train/test_currentTime
writer_path = result_root_path / 'runs' / group
writer_filename = '{0}_{1}_{2}_{3}'.format(
    opt.model_type, current_time, year, getHostName())  # modelName_currentTime_hostName

# Parameters for dataset
dataset_path = {
    'root_path': root_path,  #old was dataset_root_path
    'meta_filepath': meta_filepath,
    'train_filepath': train_filepath,
    'test_filepath': test_filepath
}

# Parameters for solver
model = {
    'model_type': opt.model_type,
    'pretrained': opt.pretrained,
    'outdim': opt.outdim,
    'resume': opt.resume,
    'loading_epoch': opt.loading_epoch,
    'total_epochs': opt.total_epochs,
    'model_path': model_path,
    'model_filename': model_filename,
    'save': opt.save_model,
    'gpu': gpu,
    'feature_extract': opt.feature_extract,
    'manual_seed': opt.manual_seed
}

optimizer = {
    'optim_type': opt.optim_type,
    'resume': opt.resume,
    'lr': opt.lr,
    'weight_decay': opt.weight_decay,
    'weighted_loss': opt.weighted_loss
}

scheduler = {
    'use': opt.scheduler,
    'step_size': opt.step_size,
    'milestones': [20, 40],
    'gamma': opt.gamma
}

logging = {
    'log_path': log_path,
    'log_filename': log_filename,
    'log_level': 20,  # 20 == level (logging.INFO)
}

writer = {'writer_path': writer_path, 'writer_filename': writer_filename}

# Log config
makeSubdir(logging['log_path'])
logger = set_logging(logging['log_path'] / logging['log_filename'],
                     logging['log_level'])


# Log model, optim information
printArgs(logger, vars(opt))
printArgs(logger, {'batch_size': bsize})

# Preprocessing transforms: data augmentation

#means = [123./255., 150./255., 86./255.] #pm
#stds = [34./255., 34./255., 45./255.] #pm

#means = [147./255., 164./255., 117./255.] #dm
#stds = [39./255., 39./255., 49./255.] #dm

means = opt.means
stds = opt.stds

if opt.seg_dataset:
    # means = [123./255., 150./255., 86./255.] #pm
    # stds = [34./255., 34./255., 45./255.] #pm

    #means = [147. / 255., 164. / 255., 117. / 255.]  # dm
    #stds = [39. / 255., 39. / 255., 49. / 255.]  # dm
    mean = opt.means
    stds = opt.stds

if opt.model_type == 'Inception3':
    train_augmentation = tvtrans.Compose([
        tvtrans.ToPILImage(),
        tvtrans.Resize(299),
        tvtrans.RandomHorizontalFlip(p=0.5),
        tvtrans.RandomVerticalFlip(p=0.5),
        tvtrans.RandomAffine(degrees=(0, 180), translate=(
            0.05, 0.05), scale=(0.9, 1.1)),
        # tvtrans.ColorJitter(brightness=[1.0, 1.3], contrast=[1.0, 1.3]),
        tvtrans.ToTensor(),
        tvtrans.Normalize(means, stds)
    ])
    test_transform = tvtrans.Compose([
        tvtrans.ToPILImage(),
        tvtrans.Resize(299),
        # tvtrans.ColorJitter(brightness=[1.0, 1.3], contrast=[1.0, 1.3]),
        tvtrans.ToTensor(),
        tvtrans.Normalize(means, stds)
    ])
else:
    train_augmentation = tvtrans.Compose([
        tvtrans.ToPILImage(),
        tvtrans.RandomHorizontalFlip(p=0.5),
        tvtrans.RandomVerticalFlip(p=0.5),
        tvtrans.RandomAffine(degrees=(0, 180), translate=(
            0.05, 0.05), scale=(0.9, 1.1)),
        # tvtrans.ColorJitter(brightness=[1.0, 1.3], contrast=[1.0, 1.3]),
        tvtrans.ToTensor(),
        tvtrans.Normalize(means, stds)
    ])
    test_transform = tvtrans.Compose([
        tvtrans.ToPILImage(),
        # tvtrans.ColorJitter(brightness=[1.0, 1.3], contrast=[1.0, 1.3]),
        tvtrans.ToTensor(),
        tvtrans.Normalize(means, stds)
    ])

logger.info(train_augmentation)


def worker_init_fn(worker_id): return np.random.seed(
    np.random.get_state()[1][0] + worker_id)


# Get Hyphal dataset
hyphal_train_ds = HyphalDataset(dataset_path,
                                train=True,
                                transform=train_augmentation)
# In case batch norm layer won't work on the single sample
drop_last = True if len(hyphal_train_ds) % bsize == 1 else False
hyphal_train_dl = torch.utils.data.DataLoader(hyphal_train_ds,
                                              batch_size=bsize,
                                              drop_last=drop_last,
                                              num_workers=nworker,
                                              worker_init_fn=worker_init_fn,
                                              shuffle=True)

hyphal_test_ds = HyphalDataset(dataset_path,
                               train=False,
                               transform=test_transform)
drop_last = True if len(hyphal_test_ds) % bsize == 1 else False
hyphal_test_dl = torch.utils.data.DataLoader(hyphal_test_ds,
                                             batch_size=bsize,
                                             drop_last=drop_last,
                                             num_workers=nworker,
                                             shuffle=False)

dataloader = {'train': hyphal_train_dl, 'valid': hyphal_test_dl}

if __name__ == '__main__':

    if opt.n_trials > 0:
        # Define the pruner
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=10, interval_steps=1)


        def objective(trial):
            # Define the search space for hyperparameters
            lr = trial.suggest_float('lr', 1e-7, 1e-2)
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1)
            optim_type = trial.suggest_categorical('optim_type', ['Adam', 'Adadelta', 'RMSprop', 'SGD'])

            # Set the hyperparameters in the optimizer dictionary

            optimizer['optim_type'] = optim_type
            optimizer['lr'] = lr
            optimizer['weight_decay'] = weight_decay

            print(colored('Hyperparameters for trial {trial.number}:', 'green'), optimizer)

            # Train the model using the new hyperparameters


            solver = HyphalSolver(model, dataloader, optimizer, scheduler, logger, writer)
            solver.forward()

            # Evaluate the model and return the validation accuracy
            return solver.evaluate()


        # Optimize hyperparameters using Optuna
        study_name = opt.study_name
        n_trials = opt.n_trials

        storage_url = "sqlite:///db.sqlite3"
        study = optuna.create_study(direction='maximize',
                                    study_name=study_name,
                                    pruner=pruner,
                                    storage=storage_url,
                                    )
        study.optimize(objective, n_trials)

        # Print the best hyperparameters
        print(colored('Best hyperparameters: ', 'green'), study.best_params)

        # Create a table of the top 5 hyperparameter combinations with their accuracy values
        trials_df = study.trials_dataframe(attrs=('number', 'value', 'params'))
        trials_df = trials_df.rename(columns={'value': 'accuracy'})
        trials_df = trials_df.sort_values(by='accuracy', ascending=False)
        top_5_trials = trials_df.head(5)

        # Print the top 5 hyperparameter combinations
        print("\nTop 5 hyperparameter combinations:")
        print(top_5_trials.to_string(index=False))

        # Train the model with the best hyperparameters
        optimizer['lr'] = study.best_params['lr']
        optimizer['optim_type'] = study.best_params['optim_type']
        optimizer['weight_decay'] = study.best_params['weight_decay']


    # Train the model
    solver = HyphalSolver(model, dataloader, optimizer, scheduler, logger, writer)
    solver.forward()

    # Evaluate the model and print the validation accuracy
    accuracy = solver.evaluate()
    print(f"Validation accuracy: {accuracy:.3f}%")

