### Solver (train and test)
import os
import time
import copy
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

from recorder import Recorder
from utils import makeSubdir, logInfoWithDot, timeSince, init_model, init_optimizer

from termcolor import colored
import optuna

if __name__ == "__main__":
    from utils import init_model, load_model, parse_model, plot_confusion_matrix


class Solver():
    def __init__(self, model, dataloader, optimizer, scheduler, logger, writer, trial=None):
        """
        Args:
            model:         Model params
            dataloader:    Dataloader params
            optimizer:     Optimizer params
            scheduler:     Scheduler params
            logger:        Logger params
            writer:        Tensorboard writer params
        """
        self.model = init_model(model)
        self.pretrained = model['pretrained']
        self.loading_epoch = model['loading_epoch']
        self.total_epochs = model['total_epochs']
        self.model_path = model['model_path']
        self.model_filename = model['model_filename']
        self.save = model['save']
        self.patience = model['patience']
        self.model_name = model['model_type']
        self.model_fullpath = str(self.model_path / self.model_filename)
        self.max_grad_norm = float(optimizer.get('max_grad_norm', 0.0) or 0.0)  # 0.0 means disabled
        self.trial = trial
        self.is_trial_run = trial is not None


        self.init_random_seed = model['manual_seed']

        # Best model
        self.is_best = False
        self.best_model = None
        self.best_acc = 0
        self.best_optim = None
        self.best_epoch = 0
        self.best_model_filepath = str(self.model_path / 'best_model_checkpoint.pth.tar')

        # Logger
        self.logger = logger

        # Loss
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        self.trainloader = dataloader['train']
        self.validloader = dataloader['valid']

        # Loss
        label_smoothing = float(optimizer.get('label_smoothing', 0.0))

        # Writer: Default path is runs/CURRENT_DATETIME_HOSTNAME
        writer_fullname = writer['writer_path'] / writer['writer_filename']
        self.writer = SummaryWriter(log_dir=str(writer_fullname))
        
        if model.get('gpu', False) and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model.to(self.device)
            # GradScaler: handle both new and old APIs
            try:
                # PyTorch ≥ 2.3: lives under torch.amp and accepts positional device string
                self.scaler = torch.amp.GradScaler('cuda')
            except Exception:
                # Older API
                self.scaler = torch.cuda.amp.GradScaler()
            logInfoWithDot(self.logger, "USING GPU")
        
        elif model.get('mps', False) and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.model = self.model.to(self.device)
            self.scaler = None  # no grad scaling on MPS
            logInfoWithDot(self.logger, "USING MPS")
        
        else:
            self.device = torch.device('cpu')
            self.scaler = None
            logInfoWithDot(self.logger, "USING CPU")
            
            
        checkpoint = None
        if model['resume']:
            load_model_path = self.model_fullpath.format(self.model_name, self.loading_epoch)
            checkpoint = torch.load(load_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logInfoWithDot(self.logger, "LOADING MODEL FINISHED")

        if optimizer['weighted_loss'] and hasattr(self.trainloader.dataset, 'labels'):
            labels_np = self.trainloader.dataset.labels
            labels_np = np.asarray(labels_np).reshape(-1).astype(np.int64)  # <-- FLATTEN
            counts = np.bincount(labels_np, minlength=model['outdim']).astype(np.float32)
            inv = counts.sum() / (counts + 1e-8)
            weights = torch.tensor(inv / inv.mean(), device=self.device, dtype=torch.float32)
        else:
            weights = None
        self.criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        
        # Optimizer
        self.optimizer = init_optimizer(optimizer, self.model)  

        # Scheduler
        self.scheduler = None
        if scheduler['use']:
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=scheduler['milestones'],
                gamma=scheduler['gamma'])
            # self.scheduler = optim.lr_scheduler.StepLR(
            #     self.optimizer,
            #     step_size=scheduler['step_size'],
            #     gamma=scheduler['gamma'])

        # Resume optimizer/scheduler/scaler state (if checkpoint exists)
        if optimizer.get('resume', False) and checkpoint is not None:
            if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logInfoWithDot(self.logger, "LOADING OPTIMIZER FINISHED")
        
            if self.scheduler and ('scheduler_state_dict' in checkpoint) and (checkpoint['scheduler_state_dict'] is not None):
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logInfoWithDot(self.logger, "LOADING SCHEDULER FINISHED")
                except Exception as e:
                    self.logger.warning(f"Could not load scheduler state: {e}")
        
            if self.scaler and ('scaler_state_dict' in checkpoint) and (checkpoint['scaler_state_dict'] is not None):
                try:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    logInfoWithDot(self.logger, "LOADING AMP SCALER FINISHED")
                except Exception as e:
                    self.logger.warning(f"Could not load AMP scaler state: {e}")
        
        # Evaluation
        self.train_recorder = Recorder('train')
        self.test_recorder = Recorder('test')

        # Timer
        self.start_time = time.time()

    def train_one_epoch(self, ep):
        pass

    def test_one_epoch(self, ep):
        pass

    def forward(self):
        start_epoch = self.loading_epoch + 1
        end_epoch = start_epoch + self.total_epochs
        for ep in range(start_epoch, end_epoch):
            np.random.seed(self.init_random_seed + ep)
            torch.manual_seed(self.init_random_seed + ep)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.init_random_seed + ep)
    
            self.train_one_epoch(ep)
            val_loss, val_acc = self.test_one_epoch(ep)
    
            # Step LR (epoch-level schedulers like MultiStep are fine here)
            if self.scheduler:
                self.scheduler.step()
                # Log LR to TB for visibility
                cur_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('LR', cur_lr, ep)
                
            # --- Optuna: report & possibly prune on *accuracy* since study direction is 'maximize'
            if self.trial is not None:
                self.trial.report(val_acc, step=ep)
                if self.trial.should_prune():
                    self.logger.info(f"Trial pruned at epoch {ep} (val_acc={val_acc:.3f})")
                    raise optuna.TrialPruned()
    
            # Always save "last"
            if self.save and not self.is_trial_run:
                makeSubdir(self.model_path)
                last_path = str(self.model_path / 'last_model_checkpoint.pth.tar')
                torch.save({
                    'epoch': ep,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': (self.scheduler.state_dict() if self.scheduler else None),
                    'scaler_state_dict': (self.scaler.state_dict() if self.scaler else None),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, last_path)
                logInfoWithDot(self.logger, f"SAVED LAST to {last_path}")
    
            # Check improvement + save "best" when improved
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_acc = val_acc
                self.patience_counter = 0
                if self.save:
                    makeSubdir(self.model_path)  # <-- ensure path exists
                    best_path = self.best_model_filepath
                    torch.save({
                        'epoch': ep,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': (self.scheduler.state_dict() if self.scheduler else None),
                        'scaler_state_dict': (self.scaler.state_dict() if self.scaler else None),
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                    }, best_path)
                    logInfoWithDot(self.logger, f"SAVED BEST to {best_path}")
            else:
                self.patience_counter += 1
    
            # Early stop if no improvement for N epochs
            if self.patience_counter >= self.patience:
                self.logger.info(
                    f"Early stopping at epoch {ep}. Best loss={self.best_loss:.6f}, acc={self.best_acc:.3f}%"
                )
                break

            
class HyphalSolver(Solver):
    def train_one_epoch(self, ep, log_interval=50):
        self.model.train()
        self.train_recorder.reset()
        lr = self.optimizer.param_groups[0]['lr']
    
        for i, (images, labels) in enumerate(self.trainloader, 0): 
            images = images.to(self.device, dtype=torch.float, non_blocking=True)
            labels = labels.to(self.device, dtype=torch.long, non_blocking=True)
            #autocast_device = 'cuda' if self.device.type == 'cuda' else 'cpu'
            amp_enabled = self.device.type in ('cuda', 'mps') and self.scaler is not None
            with torch.amp.autocast(device_type=self.device.type, enabled=amp_enabled): 
                if not self.pretrained and self.model_name == 'GoogleNet':
                    preds, aux2, aux1 = self.model(images)
                    loss1 = self.criterion(preds, labels)
                    loss2 = self.criterion(aux1, labels)
                    loss3 = self.criterion(aux2, labels)
                    loss = loss1 + 0.3 * (loss2 + loss3)
                elif self.model_name == 'Inception3':
                    preds, aux = self.model(images)
                    loss1 = self.criterion(preds, labels)
                    loss2 = self.criterion(aux, labels)
                    loss = loss1 + 0.4 * loss2
                else:
                    preds = self.model(images)
                    loss = self.criterion(preds, labels)
    
            self.train_recorder.update(preds, labels, loss.item())
    
            self.optimizer.zero_grad(set_to_none=True)
            if self.device.type == 'cuda' and self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.max_grad_norm > 0.0:
                    # Unscale before clipping when using AMP
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
    
            if i % log_interval == 0:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLearning Rate: {}\tLoss: {:.6f}\tTime Usage:{:.8}'
                    .format(ep, i * len(images), len(self.trainloader.dataset),
                            100. * i / len(self.trainloader), lr, loss.item(),
                            timeSince(self.start_time)))
    
            if i == len(self.trainloader) - 1:
                acc = 100.0 * self.train_recorder.correct / self.train_recorder.total
                self.logger.info('Loss on {0} train images: {1:.6f}'.format(self.train_recorder.total, self.train_recorder.loss))
                self.logger.info('Accuracy on {0} train images: {1:.3f}%'.format(self.train_recorder.total, acc))
                self.writer.add_scalar('Loss/train', self.train_recorder.loss, ep)
                self.writer.add_scalar('Accuracy/train', acc, ep)

    def test_one_epoch(self, ep):
        self.model.eval()
        self.test_recorder.reset()
        total_loss, total = 0.0, 0
        with torch.no_grad():
            for images, labels in self.validloader:
                images = images.to(self.device, dtype=torch.float, non_blocking=True)
                labels = labels.to(self.device, dtype=torch.long, non_blocking=True)
                #autocast_device = 'cuda' if self.device.type == 'cuda' else 'cpu'
                amp_enabled = self.device.type in ('cuda', 'mps')
                with torch.amp.autocast(device_type=self.device.type, enabled=amp_enabled):
                    preds = self.model(images)
                    loss = self.criterion(preds, labels)
                total_loss += loss.item() * images.size(0)
                self.test_recorder.update(preds, labels, loss.item())
                total += images.size(0)

        avg_loss = total_loss / max(total, 1)
        acc = float(100.0 * self.test_recorder.correct / max(self.test_recorder.total, 1))
        self.writer.add_scalar('Loss/val', avg_loss, ep)
        self.writer.add_scalar('Accuracy/val', acc, ep)
        self.logger.info(f'Val loss: {avg_loss:.6f} | Val acc: {acc:.3f}%')
        
        return avg_loss, acc


    def evaluate(self):
        self.model.eval()
        self.test_recorder.reset()
    
        with torch.no_grad():
            for _, (images, labels) in enumerate(self.validloader, 0):
                images = images.to(self.device, dtype=torch.float)
                labels = labels.to(self.device, dtype=torch.long)
                preds = self.model(images)
                loss = self.criterion(preds, labels)
                self.test_recorder.update(preds, labels, loss.item())
    
        accuracy = np.float64(100.0 * self.test_recorder.correct / self.test_recorder.total)
        self.logger.info('Loss on {0} val images: {1:.6f}'.format(self.test_recorder.total, self.test_recorder.loss))
        self.logger.info('Accuracy on {0} val images: {1:.3f}%'.format(self.test_recorder.total, accuracy))
        return accuracy
