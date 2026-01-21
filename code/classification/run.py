import os
import argparse
import numpy as np
from pathlib import Path
from termcolor import colored
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, GroupKFold

import torch
import torchvision.transforms as tvtrans

from utils import (getTimestamp, getHostName, makeSubdir,
                   logInfoWithDot, printArgs, set_logging)
from torch.utils.data import DataLoader
from dataloader import HyphalDataset
from solver import HyphalSolver

import optuna
import pandas as pd

parser = argparse.ArgumentParser()
# Model
parser.add_argument('--model_type', default='GoogleNet', help='model used for training', choices=['GoogleNet', 'ResNet', 'SqueezeNet', 'DenseNet', 'VGG', 'AlexNet', 'Inception3', 'EfficientNetV2M','EfficientNetB4','EfficientNetV2S'])
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
parser.add_argument('--means', type=float, nargs='+', default=[0.504, 0.604, 0.361],
                    help='List of means for each channel')
parser.add_argument('--stds', type=float, nargs='+', default=[0.144, 0.142, 0.192],
                    help='List of standard deviations for each channel')
parser.add_argument('--patience', type=int, default=15, help='early stopping patience')



# Optimizer
parser.add_argument('--optim_type', default='Adam', help='optimizer used for training', choices=['Adam', 'Adadelta', 'RMSprop', 'SGD', 'AdamW'])
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for optimzer')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay for optimizer')
parser.add_argument('--weighted_loss', action='store_true', help='weighted loss')
parser.add_argument('--max_grad_norm', type=float, default=None,
                    help='Clip gradients to this L2 norm (None to disable)')
parser.add_argument('--label_smoothing', type=float, default=0.0)

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
parser.add_argument('--test_hdf5', type=str, default="test.hdf5", help='test hdf5 file')
parser.add_argument('--train_hdf5', type=str, default="train.hdf5", help='train hdf5 file')
parser.add_argument('--qtl_partition_idx', type=str, help='qtl partition to be used')
parser.add_argument('--seg_idx', type=str, help='segmentation index to be used in the cross-validation of the deep learning training script')
parser.add_argument('--demo_dataset', action='store_true', help='use balanced dataset')
parser.add_argument('--seg_dataset', action='store_true', help='use randomized dataset')
parser.add_argument('--aug_dataset', action='store_true', help='use augmented dataset')
parser.add_argument('--cross_validation', action='store_true', help='use cross validation dataset')

# Fortuna
parser.add_argument('--n_trials', type=int, default=25, help='trials for fortuna')
parser.add_argument('--study_name', type=str, default='DownyTrial', help='name of study for fortuna')
parser.add_argument('--cv_folds', type=int, default=5)
parser.add_argument('--cv_group_col', type=str, default=None,
                    help='Optional: metadata column name to group by (e.g., plant_id, image_id, genotype)')



opt = parser.parse_args()

np.random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.manual_seed)

if opt.cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda_device)
    assert torch.cuda.is_available(), "CUDA requested but not available."
    gpu = torch.device("cuda")
elif opt.mps:
    assert torch.backends.mps.is_available(), "MPS requested but not available."
    gpu = torch.device("mps")
else:
    gpu = torch.device("cpu")

# Basic configuration
root_path = Path(opt.root_path)
result_root_path = root_path / 'results'
dataset_root_path = root_path / 'data'
test_hdf5_filename = opt.test_hdf5
train_hdf5_filename = opt.train_hdf5

if opt.demo_dataset:
    meta_filepath = 'metadata.csv'
    train_filepath = 'demo_train_set.hdf5'
    test_filepath = 'demo_val_set.hdf5'
    group = 'demo'
    
elif opt.aug_dataset:
    meta_filepath = 'metadata.csv'
    train_filepath = 'train_set_aug.hdf5'
    test_filepath = 'test_set_aug.hdf5'
    group = 'aug'                
else:
    meta_filepath = 'metadata.csv'
    train_filepath = opt.train_hdf5
    test_filepath = opt.test_hdf5
    group = 'asabe_journal'

if opt.cross_validation:
    group = 'cls_cv'
elif opt.qtl_partition_idx:
    dataset_root_path = dataset_root_path / \
        'qtl_partition_test' / f'partition_ratio_{opt.qtl_partition_idx}'
    group = 'qtl_partition'
elif opt.seg_dataset:
    dataset_root_path = dataset_root_path
    group = 'cls_seg_dataset'
    
if opt.cross_validation and not opt.test_date:
    raise ValueError("--test_date is required when --cross_validation is set.")

#print("Dataset path:", dataset_root_path)

# Dataloader
bsize = opt.bsize
nworker = opt.nworker

# Model file format and location configuration
#year = '2024' # this is for labeling the model
year = str(datetime.now().year)  # Convert to string for formatting (remove above line if this works)
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
    'patience': opt.patience,
    'save': opt.save_model,
    'gpu': opt.cuda,   # bool: user wants CUDA
    'mps': opt.mps,    # bool: user wants MPS
    'feature_extract': opt.feature_extract,
    'manual_seed': opt.manual_seed
}

optimizer = {
    'optim_type': opt.optim_type,
    'resume': opt.resume,
    'lr': opt.lr,
    'weight_decay': opt.weight_decay,
    'weighted_loss': opt.weighted_loss,
    'max_grad_norm': opt.max_grad_norm,
    'label_smoothing': opt.label_smoothing

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

# Log config
makeSubdir(logging['log_path'])
logger = set_logging(logging['log_path'] / logging['log_filename'],
                     logging['log_level'])


# Log model, optim information
printArgs(logger, vars(opt))
printArgs(logger, {'batch_size': bsize})

# Preprocessing transforms: data augmentation
means = opt.means
stds = opt.stds

if opt.model_type == 'Inception3':
    train_augmentation = tvtrans.Compose([
        tvtrans.ToPILImage(),
        tvtrans.Resize(299),
        tvtrans.RandomHorizontalFlip(p=0.5),
        tvtrans.RandomVerticalFlip(p=0.5),
        tvtrans.RandomAffine(degrees=(0, 180), translate=(
            0.05, 0.05), scale=(0.9, 1.1)),
        tvtrans.ColorJitter(brightness=(1.0, 1.3), contrast=(1.0, 1.3)),
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
        tvtrans.ColorJitter(brightness=[1.0, 1.3], contrast=[1.0, 1.3]),
        tvtrans.ToTensor(),
        tvtrans.Normalize(means, stds)
    ])
    test_transform = tvtrans.Compose([
        tvtrans.ToPILImage(),
        # tvtrans.ColorJitter(brightness=[1.0, 1.3], contrast=[1.0, 1.3]),
        tvtrans.ToTensor(),
        tvtrans.Normalize(means, stds)
    ])

logger.info("train augmentations:\n%s", train_augmentation)

def worker_init_fn(worker_id): return np.random.seed(
    np.random.get_state()[1][0] + worker_id)


# Get Hyphal dataset
hyphal_train_ds = HyphalDataset(dataset_path,
                                train=True,
                                transform=train_augmentation)
# In case batch norm layer won't work on the single sample

hyphal_train_dl = torch.utils.data.DataLoader(
    hyphal_train_ds,
    batch_size=bsize,
    shuffle=True,
    drop_last=True,
    num_workers=nworker,
    worker_init_fn=worker_init_fn,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(nworker > 0),
    prefetch_factor=2, 
)

hyphal_test_ds = HyphalDataset(dataset_path,
                               train=False,
                               transform=test_transform)
hyphal_test_dl = torch.utils.data.DataLoader(
    hyphal_test_ds,
    batch_size=bsize,
    shuffle=False,
    drop_last=False,
    num_workers=nworker,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(nworker > 0),
    prefetch_factor=2 if nworker > 0 else None,
)

dataloader = {'train': hyphal_train_dl, 'valid': hyphal_test_dl}

writer = {'writer_path': writer_path, 'writer_filename': writer_filename}

def build_dataloader_from_dataset_path(dataset_path_local):
    train_ds = HyphalDataset(dataset_path_local, train=True, transform=train_augmentation)
    valid_ds = HyphalDataset(dataset_path_local, train=False, transform=test_transform)

    train_dl = DataLoader(
        train_ds,
        batch_size=opt.bsize,
        shuffle=True,
        drop_last=True,
        num_workers=opt.nworker,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(opt.nworker > 0),
        prefetch_factor=2 if opt.nworker > 0 else None,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=opt.bsize,
        shuffle=False,
        drop_last=False,
        num_workers=opt.nworker,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(opt.nworker > 0),
        prefetch_factor=2 if opt.nworker > 0 else None,
    )
    return {'train': train_dl, 'valid': valid_dl}
    
def dataset_path_for_fold(base_dataset_path, fold_idx: int):
    dp = dict(base_dataset_path)

    dp['train_filepath'] = f"cv/{opt.test_date}/cv{fold_idx}/train_{fold_idx}.hdf5"
    dp['test_filepath']  = f"cv/{opt.test_date}/cv{fold_idx}/val_{fold_idx}.hdf5"

    return dp


if __name__ == "__main__":

    # =========================
    # OPTUNA path
    # =========================
    if opt.n_trials > 0:

        if opt.cross_validation:
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=opt.cv_folds,
                reduction_factor=2
            )
        else:
            pruner = optuna.pruners.NopPruner()

        def objective(trial):
            # Per-trial writer
            trial_writer_fn = f"{opt.model_type}_{current_time}_trial{trial.number}_{year}_{getHostName()}"
            trial_writer = {'writer_path': writer_path, 'writer_filename': trial_writer_fn}

            # ---- suggest hyperparams ----
            optim_type = trial.suggest_categorical("optim_type", ["AdamW", "Adadelta", "SGD"])
            lr = trial.suggest_float("lr", 1e-5, 3e-2, log=True)
            label_smoothing = trial.suggest_float("label_smoothing", 0.00, 0.10, step=0.01)

            opt_extras = {}
            if optim_type == "SGD":
                weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
                opt_extras["momentum"] = trial.suggest_float("momentum", 0.85, 0.98)
                opt_extras["nesterov"] = trial.suggest_categorical("nesterov", [False, True])
            elif optim_type == "AdamW":
                weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
                opt_extras["beta1"] = trial.suggest_float("beta1", 0.85, 0.99)
                opt_extras["beta2"] = 0.999
            else:  # Adadelta
                wd_choice = trial.suggest_categorical("wd_choice", ["zero", "small"])
                weight_decay = 0.0 if wd_choice == "zero" else trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
                opt_extras["rho"] = trial.suggest_float("rho", 0.85, 0.99)

            trial_optimizer = dict(optimizer)
            trial_optimizer.update({
                "optim_type": optim_type,
                "lr": lr,
                "weight_decay": weight_decay,
                "label_smoothing": label_smoothing,
                **opt_extras,
            })

            trial_model = dict(model)
            trial_model["save"] = False  # speed up trials

            solver = None
            
            print(colored('Hyperparameters for trial {trial.number}:', 'green'), optimizer)

            try:
                if opt.cross_validation:
                    fold_f1s = []

                    for fold_idx in range(opt.cv_folds):
                        trial_writer_fold = dict(trial_writer)
                        trial_writer_fold["writer_filename"] = f"{trial_writer_fn}_fold{fold_idx}"

                        dp_fold = dataset_path_for_fold(dataset_path, fold_idx)
                        dl_fold = build_dataloader_from_dataset_path(dp_fold)

                        solver = HyphalSolver(trial_model, dl_fold, trial_optimizer, scheduler, logger, trial_writer_fold)
                        solver.forward()

                        m = solver.evaluate()              # dict
                        fold_f1 = float(m["f1"])           # 0–1
                        fold_f1s.append(fold_f1)

                        running_mean = float(np.mean(fold_f1s))
                        trial.report(running_mean, step=fold_idx)

                        if trial.should_prune():
                            raise optuna.TrialPruned()

                        del solver
                        solver = None
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    return float(np.mean(fold_f1s))        # Optuna "value" = mean CV F1 (0–1)

                else:
                    
                    solver = HyphalSolver(trial_model, dataloader, trial_optimizer, scheduler, logger, trial_writer)
                    solver.forward()

                    m = solver.evaluate()
                    val_f1 = float(m["f1"])                # 0–1

                    trial.report(val_f1, step=model["total_epochs"])
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                    return val_f1

            finally:
                if solver is not None:
                    del solver
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        study = optuna.create_study(
            direction="maximize",
            study_name=opt.study_name,
            pruner=pruner,
            storage="sqlite:///db.sqlite3",
            load_if_exists=True
        )
        study.optimize(objective, n_trials=opt.n_trials)

        print(colored("Best hyperparameters:", "green"), study.best_params)

        trials_df = study.trials_dataframe(attrs=("number", "value", "params"))
        trials_df = trials_df.rename(columns={"value": "mean_cv_f1"})  # 0–1
        trials_df = trials_df.sort_values(by="mean_cv_f1", ascending=False)
        print("\nTop 5 hyperparameter combinations (by mean CV F1):")
        print(trials_df.head(5).to_string(index=False))

        # apply best params to optimizer dict
        best = study.best_params
        optimizer["optim_type"] = best["optim_type"]
        optimizer["lr"] = best["lr"]
        optimizer["label_smoothing"] = best.get("label_smoothing", optimizer.get("label_smoothing", 0.0))

        if best["optim_type"] == "SGD":
            optimizer["weight_decay"] = best["weight_decay"]
            optimizer["momentum"] = best["momentum"]
            optimizer["nesterov"] = best["nesterov"]
        elif best["optim_type"] == "AdamW":
            optimizer["weight_decay"] = best["weight_decay"]
            optimizer["beta1"] = best.get("beta1", 0.9)
            optimizer["beta2"] = best.get("beta2", 0.999)
        else:  # Adadelta
            optimizer["weight_decay"] = best.get("weight_decay", 0.0)
            if "rho" in best:
                optimizer["rho"] = best["rho"]

        # optional final run (only if not CV mode)
        if not opt.cross_validation:
            final_writer = {
                "writer_path": writer_path,
                "writer_filename": f"{opt.model_type}_{current_time}_best_{year}_{getHostName()}"
            }
            
            
            solver = HyphalSolver(model, dataloader, optimizer, scheduler, logger, final_writer)
            solver.forward()
            m = solver.evaluate()

            print(
                f"Validation F1: {100.0*m['f1']:.3f}% | "
                f"Accuracy: {m['accuracy']:.3f}% | "
                f"Precision: {100.0*m['precision']:.3f}% | "
                f"Recall: {100.0*m['recall']:.3f}%"
            )

    # =========================
    # Non-Optuna path
    # =========================
    else:
        writer = {"writer_path": writer_path, "writer_filename": writer_filename}

        solver = HyphalSolver(model, dataloader, optimizer, scheduler, logger, writer)
        solver.forward()
        m = solver.evaluate()

        print(
            f"Validation F1: {100.0*m['f1']:.3f}% | "
            f"Accuracy: {m['accuracy']:.3f}% | "
            f"Precision: {100.0*m['precision']:.3f}% | "
            f"Recall: {100.0*m['recall']:.3f}%"
        )
