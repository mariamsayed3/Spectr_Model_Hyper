#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script with DATA LEAKAGE
- Uses validation set for both model selection AND final evaluation
- This leads to optimistically biased results
"""
import os
import torch
import torch.nn as nn
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                           roc_curve, auc, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

from spectr import ModifiedSpecTr
from torch import optim
from torch.utils.data import DataLoader
from local_utils.tools import save_dict
from local_utils.seed_everything import seed_reproducer
from tqdm import tqdm
from Data_Generate import Data_Generate_Cho
from argument import Transform
from local_utils.misc import AverageMeter
from local_utils.tools import EarlyStopping
from local_utils.metrics import iou, dice, multi_iou
from segmentation_models_pytorch.utils.losses import DiceLoss
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')


class MultiClassDiceLoss(nn.Module):
    """Custom Dice Loss that works with multi-class segmentation and ignore_index"""
    def __init__(self, ignore_index=0, smooth=1e-6):
        super(MultiClassDiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # Get number of classes from inputs
        num_classes = inputs.size(1)
        
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Apply softmax to inputs
        inputs = F.softmax(inputs, dim=1)
        
        # Exclude ignore_index class from loss calculation
        if self.ignore_index is not None:
            # Create mask for valid pixels
            valid_mask = (targets != self.ignore_index).float()
            valid_mask = valid_mask.unsqueeze(1).expand_as(inputs)
            
            # Apply mask to both inputs and targets
            inputs = inputs * valid_mask
            targets_one_hot = targets_one_hot * valid_mask
        
        # Calculate Dice coefficient for each class
        intersection = (inputs * targets_one_hot).sum(dim=(0, 2, 3))
        union = inputs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))
        
        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return mean Dice loss (excluding background if ignore_index=0)
        if self.ignore_index == 0:
            # Skip background class
            return 1.0 - dice[1:].mean()
        else:
            return 1.0 - dice.mean()


# Dynamic class weighting for uneven class distribution
class DynamicWeightedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5, ignore_index=0, alpha=0.1):
        super(DynamicWeightedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice_loss = MultiClassDiceLoss(ignore_index=ignore_index)  # Use custom implementation
        self.ignore_index = ignore_index
        self.alpha = alpha  # Smoothing factor
        
    def forward(self, input, target):
        # Get unique classes in this batch (excluding background)
        valid_mask = (target != self.ignore_index)
        if not valid_mask.any():
            return torch.tensor(0.0, device=input.device, requires_grad=True)
        
        unique_classes = torch.unique(target[valid_mask])
        
        # Calculate class frequencies
        num_classes = input.size(1)
        class_weights = torch.ones(num_classes, device=input.device)
        
        total_valid = valid_mask.sum().float()
        
        for cls in unique_classes:
            if cls < num_classes:
                cls_freq = (target[valid_mask] == cls).sum().float()
                # Inverse frequency weighting with smoothing
                class_weights[cls] = total_valid / (cls_freq + self.alpha)
        
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * unique_classes.numel()
        
        # Weighted CrossEntropy
        ce_loss = F.cross_entropy(input, target, weight=class_weights, 
                                  ignore_index=self.ignore_index)
        
        # Dice loss (no weighting needed)
        dice_loss = self.dice_loss(input, target)
        
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def calculate_metrics_per_class(y_true, y_pred, num_classes, ignore_index=0):
    """Calculate IoU, Dice, Precision, Recall, F1 for each class (excluding background)"""
    metrics = {
        'iou': {},
        'dice': {},
        'precision': {},
        'recall': {},
        'f1': {}
    }
    
    # Create list of valid classes (excluding background)
    valid_classes = [c for c in range(num_classes) if c != ignore_index]
    
    for cls in valid_classes:
        # Create binary masks for current class
        true_mask = (y_true == cls).astype(np.float32)
        pred_mask = (y_pred == cls).astype(np.float32)
        
        # Calculate metrics
        if true_mask.sum() == 0 and pred_mask.sum() == 0:
            # Both masks are empty
            metrics['iou'][cls] = 1.0
            metrics['dice'][cls] = 1.0
            metrics['precision'][cls] = 1.0
            metrics['recall'][cls] = 1.0
            metrics['f1'][cls] = 1.0
        elif true_mask.sum() == 0:
            # Ground truth is empty but prediction is not
            metrics['iou'][cls] = 0.0
            metrics['dice'][cls] = 0.0
            metrics['precision'][cls] = 0.0
            metrics['recall'][cls] = 0.0
            metrics['f1'][cls] = 0.0
        else:
            # Calculate IoU
            intersection = (true_mask * pred_mask).sum()
            union = true_mask.sum() + pred_mask.sum() - intersection
            metrics['iou'][cls] = intersection / union if union > 0 else 0.0
            
            # Calculate Dice
            metrics['dice'][cls] = 2 * intersection / (true_mask.sum() + pred_mask.sum()) if (true_mask.sum() + pred_mask.sum()) > 0 else 0.0
            
            # Calculate Precision, Recall, F1
            if pred_mask.sum() > 0:
                metrics['precision'][cls] = intersection / pred_mask.sum()
            else:
                metrics['precision'][cls] = 0.0
                
            metrics['recall'][cls] = intersection / true_mask.sum()
            
            if metrics['precision'][cls] + metrics['recall'][cls] > 0:
                metrics['f1'][cls] = 2 * (metrics['precision'][cls] * metrics['recall'][cls]) / (metrics['precision'][cls] + metrics['recall'][cls])
            else:
                metrics['f1'][cls] = 0.0
    
    return metrics


def calculate_micro_macro_metrics(y_true, y_pred, num_classes, ignore_index=0):
    """Calculate micro and macro averaged metrics"""
    # Remove background class
    valid_classes = [c for c in range(num_classes) if c != ignore_index]
    
    # Per-class metrics
    per_class_metrics = calculate_metrics_per_class(y_true, y_pred, num_classes, ignore_index)
    
    # Macro averaging (simple average across classes)
    macro_metrics = {}
    for metric in ['iou', 'dice', 'precision', 'recall', 'f1']:
        values = [per_class_metrics[metric][cls] for cls in valid_classes if cls in per_class_metrics[metric]]
        macro_metrics[f'macro_{metric}'] = np.mean(values) if values else 0.0
    
    # Micro averaging (aggregate then compute)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for cls in valid_classes:
        true_mask = (y_true == cls)
        pred_mask = (y_pred == cls)
        
        tp = np.sum(true_mask & pred_mask)
        fp = np.sum(~true_mask & pred_mask)
        fn = np.sum(true_mask & ~pred_mask)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    micro_metrics = {}
    if total_tp + total_fp > 0:
        micro_metrics['micro_precision'] = total_tp / (total_tp + total_fp)
    else:
        micro_metrics['micro_precision'] = 0.0
    
    if total_tp + total_fn > 0:
        micro_metrics['micro_recall'] = total_tp / (total_tp + total_fn)
    else:
        micro_metrics['micro_recall'] = 0.0
    
    if micro_metrics['micro_precision'] + micro_metrics['micro_recall'] > 0:
        micro_metrics['micro_f1'] = 2 * (micro_metrics['micro_precision'] * micro_metrics['micro_recall']) / (micro_metrics['micro_precision'] + micro_metrics['micro_recall'])
    else:
        micro_metrics['micro_f1'] = 0.0
    
    # Micro IoU and Dice
    if total_tp + total_fp + total_fn > 0:
        micro_metrics['micro_iou'] = total_tp / (total_tp + total_fp + total_fn)
    else:
        micro_metrics['micro_iou'] = 0.0
    
    if 2 * total_tp + total_fp + total_fn > 0:
        micro_metrics['micro_dice'] = 2 * total_tp / (2 * total_tp + total_fp + total_fn)
    else:
        micro_metrics['micro_dice'] = 0.0
    
    return {**macro_metrics, **micro_metrics}, per_class_metrics


def calculate_auc_metrics(y_true, y_proba, num_classes, ignore_index=0):
    """Calculate AUC for each class and micro/macro AUC (excluding background)"""
    # Remove background class - only calculate for valid classes
    valid_classes = [c for c in range(num_classes) if c != ignore_index]
    
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_proba_flat = y_proba.reshape(-1, num_classes)
    
    # Remove background pixels completely from calculation
    valid_mask = y_true_flat != ignore_index
    y_true_flat = y_true_flat[valid_mask]
    y_proba_flat = y_proba_flat[valid_mask]
    
    auc_metrics = {}
    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}
    
    # Calculate AUC for each class (One-vs-Rest)
    for cls in valid_classes:
        # Create binary labels for current class
        y_binary = (y_true_flat == cls).astype(int)
        y_scores = y_proba_flat[:, cls]
        
        if len(np.unique(y_binary)) > 1:  # At least one positive and one negative example
            fpr, tpr, _ = roc_curve(y_binary, y_scores)
            roc_auc = auc(fpr, tpr)
            
            fpr_dict[cls] = fpr
            tpr_dict[cls] = tpr
            roc_auc_dict[cls] = roc_auc
            auc_metrics[f'auc_class_{cls}'] = roc_auc
        else:
            # All examples are of one class
            auc_metrics[f'auc_class_{cls}'] = np.nan
    
    # Calculate micro-average AUC
    # Create binary arrays for all classes
    y_true_all = []
    y_scores_all = []
    
    for cls in valid_classes:
        y_binary = (y_true_flat == cls).astype(int)
        y_scores = y_proba_flat[:, cls]
        y_true_all.extend(y_binary)
        y_scores_all.extend(y_scores)
    
    # Compute micro-average AUC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_all, y_scores_all)
    auc_metrics['micro_auc'] = auc(fpr_micro, tpr_micro)
    
    # Store for plotting
    fpr_dict['micro'] = fpr_micro
    tpr_dict['micro'] = tpr_micro
    roc_auc_dict['micro'] = auc_metrics['micro_auc']
    
    # Calculate macro-average AUC
    valid_aucs = [auc_metrics[f'auc_class_{cls}'] for cls in valid_classes 
                  if not np.isnan(auc_metrics[f'auc_class_{cls}'])]
    auc_metrics['macro_auc'] = np.mean(valid_aucs) if valid_aucs else np.nan
    
    return auc_metrics, fpr_dict, tpr_dict, roc_auc_dict


def plot_roc_curves(fpr_dict, tpr_dict, roc_auc_dict, output_path, experiment_name, epoch=None):
    """Plot ROC curves for all classes"""
    plt.figure(figsize=(10, 8))
    
    # Colors for different classes
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
    
    # Plot ROC curve for each class
    for cls, color in zip(sorted([k for k in fpr_dict.keys() if k != 'micro']), colors):
        plt.plot(fpr_dict[cls], tpr_dict[cls], color=color, lw=2,
                label=f'Class {cls} (AUC = {roc_auc_dict[cls]:.3f})')
    
    # Plot micro-average ROC curve
    plt.plot(fpr_dict['micro'], tpr_dict['micro'], color='deeppink', linestyle='--', lw=2,
            label=f'Micro-average (AUC = {roc_auc_dict["micro"]:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    if epoch is not None:
        plt.title(f'ROC Curves - Epoch {epoch}')
        filename = f'roc_curves_epoch_{epoch}.png'
    else:
        plt.title('ROC Curves - Final Model')
        filename = 'roc_curves_final.png'
    
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(os.path.join(output_path, experiment_name, filename), 
                dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_model(model, val_loader, device, num_classes, ignore_index=0):
    """Comprehensive model evaluation"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for image, label in tqdm(val_loader, desc="Evaluating"):
            image, label = image.to(device), label.to(device)
            
            # Forward pass
            output = model(image)
            
            # Get probabilities
            probs = F.softmax(output, dim=1)
            
            # Get predictions
            pred = torch.argmax(output, dim=1)
            
            # Move to CPU and convert to numpy
            pred_np = pred.cpu().numpy()
            label_np = label.cpu().numpy()
            probs_np = probs.cpu().numpy()
            
            # Ensure both pred and label are 2D for metric calculation
            pred_np = pred_np.squeeze()
            label_np = label_np.squeeze()
            probs_np = probs_np.squeeze(0) if probs_np.ndim == 4 else probs_np
            
            all_preds.append(pred_np)
            all_labels.append(label_np)
            all_probs.append(probs_np)
    
    # Concatenate all results
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    # Calculate comprehensive metrics
    metrics, per_class_metrics = calculate_micro_macro_metrics(all_labels, all_preds, num_classes, ignore_index)
    auc_metrics, fpr_dict, tpr_dict, roc_auc_dict = calculate_auc_metrics(all_labels, all_probs, num_classes, ignore_index)
    
    return {**metrics, **auc_metrics}, per_class_metrics, (fpr_dict, tpr_dict, roc_auc_dict)


def main(args):
    seed_reproducer(42)
    
    # Parse arguments
    root_path = args.root_path
    dataset_divide = args.dataset_divide
    batch = args.batch
    experiment_name = args.experiment_name
    output_path = args.output
    epochs = args.epochs
    cutting = args.cutting
    worker = args.worker
    device = torch.device(args.device)
    classes = args.classes
    
    print(f"Using device: {device}")
    print("="*50)
    print("⚠️  WARNING: DATA LEAKAGE PRESENT ⚠️")
    print("This script uses validation set for both:")
    print("1. Model selection (early stopping)")
    print("2. Final evaluation")
    print("This leads to optimistically biased results!")
    print("="*50)
    
    # Create output directory
    if not os.path.exists(os.path.join(output_path, experiment_name)):
        os.makedirs(os.path.join(output_path, experiment_name))
    save_dict(os.path.join(output_path, experiment_name, 'args.csv'), args.__dict__)
    
    # Load dataset split
    with open(dataset_divide, 'r') as f:
        dataset_dict = json.load(f)
    
    # Simple train/validation split (no folds)
    if 'train' in dataset_dict and 'val' in dataset_dict:
        train_files = dataset_dict['train']
        val_files = dataset_dict['val']
    else:
        # Fallback: use fold1 as train, fold2 as val
        train_files = dataset_dict['fold1'] + dataset_dict['fold2'] + dataset_dict['fold3']
        val_files = dataset_dict['fold4']
    
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print("Note: Same validation set used for model selection AND final evaluation")
    
    # Data paths for bio dataset
    train_images_path = [os.path.join(root_path, f"gene_expre_matrix_{i}.npy") for i in train_files]
    train_masks_path = [os.path.join(root_path, f"label_matrix_{i}.npy") for i in train_files]
    val_images_path = [os.path.join(root_path, f"gene_expre_matrix_{i}.npy") for i in val_files]
    val_masks_path = [os.path.join(root_path, f"label_matrix_{i}.npy") for i in val_files]
    
    # Data augmentation
    transform = Transform(Rotate_ratio=0.2, Flip_ratio=0.2)
    
    # Create datasets
    train_db = Data_Generate_Cho(
        train_images_path, train_masks_path, 
        cutting=cutting, transform=transform, 
        channels=None, outtype='3d', envi_type='img',
        multi_class=classes
    )
    train_loader = DataLoader(train_db, batch_size=batch, shuffle=True, num_workers=worker)
    
    val_db = Data_Generate_Cho(
        val_images_path, val_masks_path, 
        cutting=None, transform=None,
        channels=None, outtype='3d', envi_type='img',
        multi_class=classes
    )
    val_loader = DataLoader(val_db, batch_size=1, shuffle=False, num_workers=worker)
    
    # Print data shapes
    print(f"Training data shape: {train_db[0][0].shape}")
    print(f"Training label shape: {train_db[0][1].shape}")
    
    # Create model
    model = ModifiedSpecTr(
        num_levels=4,
        f_maps=16,
        in_channels=1,
        classes=classes,
        dropout=0.1,
        dropout_att=0.1,
        activation='softmax',
        decode_choice='3D',
        use_layerscale=True,
        init_values=1.0
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-5
    )
    
    # Loss function for spatial transcriptomics
    criterion = DynamicWeightedLoss(
        ce_weight=0.7,
        dice_weight=0.3,
        ignore_index=0,
        alpha=0.1
    )
    
    # Early stopping and metrics tracking
    early_stopping = EarlyStopping(patience=20, verbose=True, 
                                   output_path=output_path, 
                                   experiment_name=experiment_name, 
                                   fold=None)
    
    # Enhanced history tracking
    history = {
        'epoch': [], 'lr': [], 'train_loss': [], 'train_iou': [],
        'val_dice': [], 'val_iou': [], 'val_count': [],
        # Macro metrics
        'macro_iou': [], 'macro_dice': [], 'macro_precision': [], 
        'macro_recall': [], 'macro_f1': [], 'macro_auc': [],
        # Micro metrics
        'micro_iou': [], 'micro_dice': [], 'micro_precision': [], 
        'micro_recall': [], 'micro_f1': [], 'micro_auc': [],
        # Per-class AUC
        **{f'auc_class_{i}': [] for i in range(1, classes)}
    }
    confusion_matrix_history = []
    per_class_metrics_history = []
    
    # Variables to store best model metrics
    best_metrics = None
    best_per_class_metrics = None
    best_roc_data = None
    best_epoch = 0
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = AverageMeter()
        train_iou = 0
        
        print(f'Epoch {epoch + 1}/{epochs}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        for idx, (image, label) in enumerate(tqdm(train_loader, desc="Training")):
            image, label = image.to(device), label.to(device)
            
            # Forward pass with masking
            out = model(image, label)  # Pass label for masking
            
            # Prepare labels for loss calculation
            label = label.squeeze(1).long()
            
            # Calculate loss
            loss = criterion(out, label)
            
            # Log class distribution occasionally
            if idx % 50 == 0:
                unique_classes = torch.unique(label[label != 0])
                print(f"  Batch {idx} classes: {unique_classes.tolist()}")
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_losses.update(loss.item())
            
            # Calculate IoU for this batch
            out_np = out.detach().cpu().numpy()
            label_np = label.cpu().numpy()
            
            # Convert predictions to class labels
            pred_np = np.argmax(out_np, axis=1)
            
            # Calculate IoU
            batch_iou = np.mean([multi_iou(pred_np[b], label_np[b]) for b in range(len(pred_np))])
            train_iou += batch_iou
        
        train_iou /= len(train_loader)
        
        # Comprehensive validation evaluation (USING VALIDATION SET)
        print("Performing validation evaluation...")
        val_metrics, per_class_metrics, roc_data = evaluate_model(model, val_loader, device, classes, ignore_index=0)
        fpr_dict, tpr_dict, roc_auc_dict = roc_data
        # Extract specific metrics for early stopping
        val_dice = val_metrics['macro_dice']
        val_iou = val_metrics['macro_iou']
        
        # Print epoch results
        print(f'Epoch {epoch + 1}:')
        print(f'  Train Loss: {train_losses.avg:.4f}, Train IoU: {train_iou:.4f}')
        print(f'  Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}')
        print(f'  Macro - Precision: {val_metrics["macro_precision"]:.4f}, Recall: {val_metrics["macro_recall"]:.4f}, F1: {val_metrics["macro_f1"]:.4f}, AUC: {val_metrics["macro_auc"]:.4f}')
        print(f'  Micro - Precision: {val_metrics["micro_precision"]:.4f}, Recall: {val_metrics["micro_recall"]:.4f}, F1: {val_metrics["micro_f1"]:.4f}, AUC: {val_metrics["micro_auc"]:.4f}')
        
        # Update history
        history['epoch'].append(epoch + 1)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['train_loss'].append(train_losses.avg)
        history['train_iou'].append(train_iou)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        
        # Add macro metrics
        history['macro_iou'].append(val_metrics['macro_iou'])
        history['macro_dice'].append(val_metrics['macro_dice'])
        history['macro_precision'].append(val_metrics['macro_precision'])
        history['macro_recall'].append(val_metrics['macro_recall'])
        history['macro_f1'].append(val_metrics['macro_f1'])
        history['macro_auc'].append(val_metrics['macro_auc'])
        
        # Add micro metrics
        history['micro_iou'].append(val_metrics['micro_iou'])
        history['micro_dice'].append(val_metrics['micro_dice'])
        history['micro_precision'].append(val_metrics['micro_precision'])
        history['micro_recall'].append(val_metrics['micro_recall'])
        history['micro_f1'].append(val_metrics['micro_f1'])
        history['micro_auc'].append(val_metrics['micro_auc'])
        
        # Add per-class AUC
        for cls in range(1, classes):
            key = f'auc_class_{cls}'
            if key in val_metrics:
                history[key].append(val_metrics[key])
            else:
                history[key].append(np.nan)
        
        # Store per-class metrics
        per_class_metrics_history.append(per_class_metrics)
        
        # Plot ROC curves every 10 epochs
        if (epoch + 1) % 10 == 0:
            fpr_dict, tpr_dict, roc_auc_dict = roc_data
            plot_roc_curves(fpr_dict, tpr_dict, roc_auc_dict, output_path, experiment_name, epoch + 1)
        
        # Confusion matrix (for compatibility and visualization)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for image, label in val_loader:
                image, label = image.to(device), label.to(device)
                output = model(image)
                pred = torch.argmax(output, dim=1)
                
                # Move to CPU
                pred_np = pred.cpu().numpy().flatten()
                label_np = label.squeeze().cpu().numpy().flatten()
                
                # Filter out background class
                valid_mask = label_np != 0
                pred_np = pred_np[valid_mask]
                label_np = label_np[valid_mask]
                
                all_preds.extend(pred_np)
                all_labels.extend(label_np)
        
        # Create confusion matrix with labels excluding class 0
        valid_labels = list(range(1, classes))
        cm = confusion_matrix(all_labels, all_preds, labels=valid_labels)
        confusion_matrix_history.append(pd.DataFrame(cm, index=valid_labels, columns=valid_labels))
        
        # Update learning rate
        scheduler.step()
        
        # Early stopping and save best model metrics
        # Store previous best score to detect if model was saved
        prev_best_score = early_stopping.best_score
        
        early_stopping(-val_dice, model, confusion_matrix_history[-1])
        history['val_count'].append(early_stopping.counter)
        
        # Save model checkpoints
        if args.save_every_epoch and (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), 
                      os.path.join(output_path, experiment_name, f'model_epoch_{epoch}.pth'))
        
        # Check if early stopping saved a new best model
        # This happens when best_score changes (either from None to a value, or improves)
        if early_stopping.best_score != prev_best_score:
            print(f"  New best model saved! (Dice: {val_dice:.4f})")
            print(f"  Best score improved from {prev_best_score} to {early_stopping.best_score}")
            
            # Save metrics for the best model (no need to save model, early stopping did it)
            best_metrics = val_metrics.copy()
            best_per_class_metrics = per_class_metrics.copy()
            fpr_dict, tpr_dict, roc_auc_dict = roc_data
            best_roc_data = (fpr_dict.copy(), tpr_dict.copy(), roc_auc_dict.copy())
            best_epoch = epoch + 1
        
        # Early stop check
        if early_stopping.early_stop:
            print(f'Early stopping at epoch {epoch + 1}')
            break
        
        # Save enhanced history
        history_pd = pd.DataFrame(history)
        history_pd.to_csv(os.path.join(output_path, experiment_name, 'training_log.csv'), index=False)
        
        # Save per-class metrics history
        per_class_df = pd.DataFrame({
            f'epoch_{i+1}': pd.Series(per_class_metrics_history[i]) 
            for i in range(len(per_class_metrics_history))
        }).transpose()
        per_class_df.to_csv(os.path.join(output_path, experiment_name, 'per_class_metrics_history.csv'))
        
        # Save confusion matrix history
        cm_pd = pd.concat(confusion_matrix_history, keys=range(1, epoch + 2))
        cm_pd.to_csv(os.path.join(output_path, experiment_name, 'confusion_matrix_history.csv'), 
                    index_label=['Epoch'], index=True)
        
        torch.cuda.empty_cache()
    
    # Use the saved best model metrics (no need to reload and re-evaluate)
    if best_metrics is not None:
        print(f"\nUsing saved metrics from best model (Epoch {best_epoch})...")
        print("⚠️  WARNING: These metrics are from the SAME validation set used for model selection!")
        print("   This leads to optimistic bias in the results.")
        
        # Use the saved metrics from the best model
        final_metrics = best_metrics
        final_per_class = best_per_class_metrics
        final_roc_data = best_roc_data
    else:
        print("\n" + "="*50)
        print("⚠️  ERROR: No best model was saved during training!")
        print("This indicates a problem with the early stopping implementation.")
        print("Early stopping should save at least one model during training.")
        print("Please check your EarlyStopping class implementation.")
        print("="*50)
        exit(1)
    
    # Save final metrics
    final_metrics_df = pd.DataFrame([final_metrics])
    final_metrics_df.to_csv(os.path.join(output_path, experiment_name, 'final_metrics.csv'), index=False)
    
    # Save final per-class metrics
    final_per_class_df = pd.DataFrame(final_per_class)
    final_per_class_df.to_csv(os.path.join(output_path, experiment_name, 'final_per_class_metrics.csv'))
    
    # Plot final ROC curves
    if final_roc_data is not None:
        fpr_dict, tpr_dict, roc_auc_dict = final_roc_data
        plot_roc_curves(fpr_dict, tpr_dict, roc_auc_dict, output_path, experiment_name)
    else:
        print("WARNING: No ROC data available for plotting final curves")
    
    # Get the final confusion matrix from the saved history (best epoch)
    best_epoch_idx = best_epoch - 1
    if best_epoch_idx < len(confusion_matrix_history):
        final_cm_df = confusion_matrix_history[best_epoch_idx]
        final_cm_df.to_csv(os.path.join(output_path, experiment_name, 'final_confusion_matrix.csv'), index=True)
    
    # Create a comprehensive summary report
    print("\nGenerating final summary report...")
    print(f"Note: All metrics exclude background class (class 0)")
    summary_report = {
        'Model': 'ModifiedSpecTr',
        'Classes': f'{classes} (excluding background)',
        'Valid Classes': f'{list(range(1, classes))}',
        'Best Epoch': best_epoch,
        'Data Split': 'Train/Validation (NO TEST SET)',
        'WARNING': 'VALIDATION SET USED FOR BOTH MODEL SELECTION AND FINAL EVALUATION - RESULTS ARE OPTIMISTICALLY BIASED!',
        'Final Macro Dice': final_metrics['macro_dice'],
        'Final Macro IoU': final_metrics['macro_iou'],
        'Final Macro Precision': final_metrics['macro_precision'],
        'Final Macro Recall': final_metrics['macro_recall'],
        'Final Macro F1': final_metrics['macro_f1'],
        'Final Macro AUC': final_metrics['macro_auc'],
        'Final Micro Dice': final_metrics['micro_dice'],
        'Final Micro IoU': final_metrics['micro_iou'],
        'Final Micro Precision': final_metrics['micro_precision'],
        'Final Micro Recall': final_metrics['micro_recall'],
        'Final Micro F1': final_metrics['micro_f1'],
        'Final Micro AUC': final_metrics['micro_auc'],
    }
    
    # Add per-class metrics to summary
    for cls in range(1, classes):
        if cls in final_per_class['dice']:
            summary_report[f'Class {cls} Dice'] = final_per_class['dice'][cls]
        if f'auc_class_{cls}' in final_metrics:
            summary_report[f'Class {cls} AUC'] = final_metrics[f'auc_class_{cls}']
    
    summary_df = pd.DataFrame([summary_report])
    summary_df.to_csv(os.path.join(output_path, experiment_name, 'summary_report.csv'), index=False)
    
    print("Training completed!")
    print(f"Best model saved at: {os.path.join(output_path, experiment_name, 'best_model.pth')}")
    print(f"Best model was from Epoch {best_epoch}")
    print(f"Final Macro Dice: {final_metrics['macro_dice']:.4f}")
    print(f"Final Macro AUC: {final_metrics['macro_auc']:.4f}")
    print("\n" + "="*50)
    print("⚠️  IMPORTANT: Results may be optimistically biased!")
    print("   Consider using a separate test set for unbiased evaluation.")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spatial Transcriptomics Training with DATA LEAKAGE')
    
    # Dataset arguments
    parser.add_argument('--root_path', '-r', type=str, required=True, 
                       help='Root path to dataset')
    parser.add_argument('--dataset_divide', '-dd', type=str, required=True,
                       help='JSON file containing train/val split')
    
    # Model arguments
    parser.add_argument('--classegit s', '-c', type=int, default=5,
                       help='Number of classes (including background)')
    
    # Training arguments
    parser.add_argument('--batch', '-b', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--cutting', '-cut', type=int, default=192,
                       help='Patch size for training')
    
    # Output arguments
    parser.add_argument('--output', '-o', type=str, default='./results',
                       help='Output directory')
    parser.add_argument('--experiment_name', '-name', type=str, required=True,
                       help='Experiment name')
    parser.add_argument('--save_every_epoch', action='store_true',
                       help='Save model every 5 epochs')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for training')
    parser.add_argument('--worker', '-nw', type=int, default=4,
                       help='Number of workers for data loading')
    
    args = parser.parse_args()
    main(args)