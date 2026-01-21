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

def confusion_from_logits(logits: torch.Tensor, y_true: torch.Tensor, num_classes: int):
    """
    logits: (N, C)
    y_true: (N,) int64
    returns: conf (C, C) where rows=true, cols=pred
    """
    y_pred = torch.argmax(logits, dim=1)
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=y_true.device)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        conf[t.long(), p.long()] += 1
    return conf


def prf_from_confusion(conf: torch.Tensor, average: str = "macro"):
    """
    conf: (C,C) on CPU or GPU
    average: 'macro' | 'weighted' | 'micro'
    returns dict: precision, recall, f1 (floats)
    """
    conf = conf.to(torch.float64)
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    support = conf.sum(dim=1)  # true counts per class

    precision_c = tp / (tp + fp + 1e-12)
    recall_c    = tp / (tp + fn + 1e-12)
    f1_c        = 2 * precision_c * recall_c / (precision_c + recall_c + 1e-12)

    if average == "macro":
        precision = precision_c.mean()
        recall = recall_c.mean()
        f1 = f1_c.mean()

    elif average == "weighted":
        w = support / (support.sum() + 1e-12)
        precision = (precision_c * w).sum()
        recall = (recall_c * w).sum()
        f1 = (f1_c * w).sum()

    elif average == "micro":
        # micro = compute global TP/FP/FN
        TP = tp.sum()
        FP = fp.sum()
        FN = fn.sum()
        precision = TP / (TP + FP + 1e-12)
        recall = TP / (TP + FN + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

    else:
        raise ValueError("average must be one of: macro, weighted, micro")

    return {
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "f1": float(f1.item()),
        "per_class_precision": precision_c.detach().cpu().numpy(),
        "per_class_recall": recall_c.detach().cpu().numpy(),
        "per_class_f1": f1_c.detach().cpu().numpy(),
        "support": support.detach().cpu().numpy(),
    }

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
        self.outdim = model['outdim']
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
        self.best_metrics = None
        self.best_f1 = -1.0
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
        end_epoch = self.total_epochs + 1

    
        for ep in range(start_epoch, end_epoch):
            np.random.seed(self.init_random_seed + ep)
            torch.manual_seed(self.init_random_seed + ep)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.init_random_seed + ep)
    
            self.train_one_epoch(ep)
    
            # Expect: val_metrics in [0,1] for precision/recall/f1
            val_loss, val_accuracy, val_metrics = self.test_one_epoch(ep)
    
            # Convert to percentage once (keep consistent everywhere)
            val_f1 = float(100.0 * val_metrics["f1"])
            val_recall = float(100.0 * val_metrics["recall"])
            val_precision = float(100.0 * val_metrics["precision"])
    
            # Step LR
            if self.scheduler:
                self.scheduler.step()
                cur_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('LR', cur_lr, ep)
    
            # --- Optuna: report/prune on F1 ---
            if self.trial is not None:
                self.trial.report(val_f1, step=ep)
                if self.trial.should_prune():
                    self.logger.info(f"Trial pruned at epoch {ep} (val_f1={val_f1:.3f})")
                    raise optuna.TrialPruned()
    
            # Always save "last" (non-trial runs only)
            if self.save and not self.is_trial_run:
                makeSubdir(self.model_path)
                last_path = str(self.model_path / 'last_model_checkpoint.pth.tar')
                torch.save({
                    'epoch': ep,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': (self.scheduler.state_dict() if self.scheduler else None),
                    'scaler_state_dict': (self.scaler.state_dict() if self.scaler else None),
    
                    # metrics
                    'val_loss': float(val_loss),
                    'val_accuracy': float(val_accuracy),          # already %
                    'val_f1': val_f1,                   # %
                    'val_recall': val_recall,           # %
                    'val_precision': val_precision,     # %
                }, last_path)
                logInfoWithDot(self.logger, f"SAVED LAST to {last_path}")
    
            # Check improvement + save "best" (use F1)
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.best_loss = float(val_loss)
                self.best_acc = float(val_accuracy)
                self.best_epoch = ep
                self.patience_counter = 0
                
                self.best_metrics = {
                    "epoch": ep,
                    "val_loss": float(val_loss),
                    "val_accuracy": float(val_accuracy),
                    "val_f1": float(val_f1),
                    "val_precision": float(val_precision),
                    "val_recall": float(val_recall),
                }
    
                if self.save and not self.is_trial_run:
                    makeSubdir(self.model_path)
                    best_path = self.model_fullpath.format(self.model_name, ep)
                    torch.save({
                        'epoch': ep,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': (self.scheduler.state_dict() if self.scheduler else None),
                        'scaler_state_dict': (self.scaler.state_dict() if self.scaler else None),
    
                        # metrics
                        'val_loss': float(val_loss),
                        'val_accuracy': float(val_accuracy),
                        'val_f1': val_f1,
                        'val_recall': val_recall,
                        'val_precision': val_precision,
                    }, best_path)
                    logInfoWithDot(self.logger, f"SAVED BEST to {best_path}")
            else:
                self.patience_counter += 1
    
            # Early stop
            if self.patience_counter >= self.patience:
                self.logger.info(
                    f"Early stopping at epoch {ep}. "
                    f"Best F1={self.best_f1:.3f}%, "
                    f"best loss={self.best_loss:.6f}, best acc={self.best_acc:.3f}% "
                    f"(best epoch={self.best_epoch})"
                )
                
                break
        
        if self.best_metrics is not None:
            self.logger.info("============================================")
            self.logger.info(
                f"✅ BEST MODEL @ epoch {self.best_metrics['epoch']} | "
                f"Val Accuracy: {self.best_metrics['val_accuracy']:.3f}% | "
                f"Val F1: {self.best_metrics['val_f1']:.3f}% | "
                f"Val Precision: {self.best_metrics['val_precision']:.3f}% | "
                f"Val Recall: {self.best_metrics['val_recall']:.3f}% | "
                f"Val Loss: {self.best_metrics['val_loss']:.6f}"
            )
            if self.save and getattr(self, "best_path", None):
                self.logger.info(f"SAVED BEST to {self.best_path}")
            self.logger.info("============================================")

            
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
                train_accuracy = 100.0 * self.train_recorder.correct / self.train_recorder.total
                self.logger.info('Loss on {0} train images: {1:.6f}'.format(self.train_recorder.total, self.train_recorder.loss))
                self.logger.info('Accuracy on {0} train images: {1:.3f}%'.format(self.train_recorder.total, train_accuracy))
                self.writer.add_scalar('Loss/train', self.train_recorder.loss, ep)
                self.writer.add_scalar('Accuracy/train', train_accuracy, ep)

    def test_one_epoch(self, ep):
        self.model.eval()
        self.test_recorder.reset()

        num_classes = self.outdim  

        total_loss, total = 0.0, 0
        conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=self.device)

        with torch.no_grad():
            for images, labels in self.validloader:
                images = images.to(self.device, dtype=torch.float, non_blocking=True)
                labels = labels.to(self.device, dtype=torch.long, non_blocking=True)

                amp_enabled = self.device.type in ('cuda', 'mps')
                with torch.amp.autocast(device_type=self.device.type, enabled=amp_enabled):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                total_loss += loss.item() * images.size(0)
                total += images.size(0)

                self.test_recorder.update(logits, labels, loss.item())
                conf += confusion_from_logits(logits, labels, num_classes)

        avg_loss = total_loss / max(total, 1)
        val_accuracy = float(100.0 * self.test_recorder.correct / max(self.test_recorder.total, 1))

        metrics = prf_from_confusion(conf, average="macro")
        f1 = 100.0 * metrics["f1"]
        recall = 100.0 * metrics["recall"]
        precision = 100.0 * metrics["precision"]

        self.writer.add_scalar('Loss/val', avg_loss, ep)
        self.writer.add_scalar('Accuracy/val', val_accuracy, ep)
        self.writer.add_scalar('F1/val', f1, ep)
        self.writer.add_scalar('Recall/val', recall, ep)
        self.writer.add_scalar('Precision/val', precision, ep)

        self.logger.info(
            f"Val loss: {avg_loss:.6f} | Val acc: {val_accuracy:.3f}% | "
            f"Val F1(macro): {f1:.3f}% | Val Recall(macro): {recall:.3f}% | "
            f"Val Precision(macro): {precision:.3f}%"
        )

        return avg_loss, val_accuracy, metrics

    def evaluate(self, ep: int | None = None, average: str = "macro"):
        """
        Evaluate on self.validloader for either binary (2-class) or multi-class (3-class).

        Returns:
            metrics_out (dict) with:
              - loss (float)
              - accuracy (float, percent)
              - precision, recall, f1 (float, 0-1)
              - per_class_precision/recall/f1, support
              - confusion (numpy array)
        """
        self.model.eval()
        self.test_recorder.reset()

        num_classes = int(self.outdim)  # <-- make sure __init__ sets self.outdim = model["outdim"]
        total_loss, total = 0.0, 0

        conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=self.device)

        with torch.no_grad():
            for images, labels in self.validloader:
                images = images.to(self.device, dtype=torch.float, non_blocking=True)
                labels = labels.to(self.device, dtype=torch.long, non_blocking=True)

                amp_enabled = self.device.type in ('cuda', 'mps')
                with torch.amp.autocast(device_type=self.device.type, enabled=amp_enabled):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                bs = images.size(0)
                total_loss += loss.item() * bs
                total += bs

                self.test_recorder.update(logits, labels, loss.item())
                conf += confusion_from_logits(logits, labels, num_classes)

        avg_loss = total_loss / max(total, 1)
        val_accuracy = float(100.0 * self.test_recorder.correct / max(self.test_recorder.total, 1))

        metrics = prf_from_confusion(conf, average=average)  # precision/recall/f1 in [0,1]
        f1 = 100.0 * metrics["f1"]
        recall = 100.0 * metrics["recall"]
        precision = 100.0 * metrics["precision"]

        # Optional: also report class-1 ("infected") metrics for binary case
        infected_metrics = {}
        if num_classes == 2:
            infected_metrics = {
                "infected_precision": float(metrics["per_class_precision"][1]),
                "infected_recall": float(metrics["per_class_recall"][1]),
                "infected_f1": float(metrics["per_class_f1"][1]),
            }

        # TensorBoard (optional)
        if ep is not None:
            self.writer.add_scalar("Loss/eval", avg_loss, ep)
            self.writer.add_scalar("Accuracy/eval", val_accuracy, ep)
            self.writer.add_scalar(f"F1_{average}/eval", f1, ep)
            self.writer.add_scalar(f"Recall_{average}/eval", recall, ep)
            self.writer.add_scalar(f"Precision_{average}/eval", precision, ep)

            if num_classes == 2:
                self.writer.add_scalar("F1_infected/eval", 100.0 * infected_metrics["infected_f1"], ep)
                self.writer.add_scalar("Recall_infected/eval", 100.0 * infected_metrics["infected_recall"], ep)
                self.writer.add_scalar("Precision_infected/eval", 100.0 * infected_metrics["infected_precision"], ep)

        # Logger
        msg = (
            f"Eval loss: {avg_loss:.6f} | Eval acc: {val_accuracy:.3f}% | "
            f"Eval F1({average}): {f1:.3f}% | Eval Recall({average}): {recall:.3f}% | "
            f"Eval Precision({average}): {precision:.3f}%"
        )
        if num_classes == 2:
            msg += (
                f" | Infected F1: {100.0 * infected_metrics['infected_f1']:.3f}%"
                f" | Infected Recall: {100.0 * infected_metrics['infected_recall']:.3f}%"
                f" | Infected Precision: {100.0 * infected_metrics['infected_precision']:.3f}%"
            )
        self.logger.info(msg)

        metrics_out = {
            "loss": float(avg_loss),
            "accuracy": float(val_accuracy),  # percent
            "precision": float(metrics["precision"]),  # 0-1
            "recall": float(metrics["recall"]),  # 0-1
            "f1": float(metrics["f1"]),  # 0-1
            "per_class_precision": metrics["per_class_precision"],
            "per_class_recall": metrics["per_class_recall"],
            "per_class_f1": metrics["per_class_f1"],
            "support": metrics["support"],
            "confusion": conf.detach().cpu().numpy(),
            **infected_metrics,
        }
        return metrics_out
