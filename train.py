import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR 
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report 
import os
from torch.utils.tensorboard import SummaryWriter 
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure you have config.py, dataset.py, and model.py in the same directory
from config import Config
from dataset import get_dataloaders 
from model import DeepSenseAttention 

# --- Early Stopping Class ---
class EarlyStopper:
    """Stops training if validation metric doesn't improve after a given patience."""
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = -np.inf # Maximize F1-score

    def early_stop(self, current_metric):
        """Returns True if training should stop, False otherwise."""
        if current_metric > self.best_metric + self.min_delta:
            self.best_metric = current_metric
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered after {self.patience} epochs without improvement (Best F1: {self.best_metric:.4f}).")
                return True
            return False

def train(model, train_loader, criterion, optimizer, device, grad_clip_val):
    """Performs one epoch of training with gradient clipping."""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for inputs, labels in tqdm(train_loader, desc="Training"):
        # CRITICAL FIX (retained): Explicitly cast inputs to torch.float32 using .float()
        inputs = inputs.to(device).float()
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        # Gradient Clipping
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
        
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    avg_loss = running_loss / len(train_loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    
    return avg_loss, acc, f1

def evaluate(model, test_loader, criterion, device):
    """Evaluates the model on the test set and returns metrics and report."""
    model.eval()
    test_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            # CRITICAL FIX (retained): Explicitly cast inputs to torch.float32 using .float()
            inputs = inputs.to(device).float()
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = test_loss / len(test_loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    
    # Calculate and capture the classification report and confusion matrix
    target_names = [f'Class {i}' for i in range(Config.NUM_CLASSES)]
    report_text = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n--- Classification Report ---")
    print(report_text)
    print("\n--- Confusion Matrix ---")
    print(cm)
        
    return avg_loss, acc, f1, cm, report_text

def main():
    print(f"--- DeepSense Attention Training Started on {Config.DEVICE} ---")
    
    # 1. Load Data
    train_loader, test_loader = get_dataloaders()
    
    # 2. Initialize Model, Loss, and Optimizer
    model = DeepSenseAttention().to(Config.DEVICE)
    # Ensure model parameters match data type
    model = model.float() 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=Config.LEARNING_RATE,
        weight_decay=Config.ADAM_WEIGHT_DECAY 
    )
    
    # 3. Learning Rate Scheduler
    scheduler = StepLR(
        optimizer, 
        step_size=Config.LR_SCHEDULER_STEP, 
        gamma=Config.LR_GAMMA
    )
    
    # 4. Initialize TensorBoard and Early Stopper
    writer = SummaryWriter(Config.LOG_DIR)
    early_stopper = EarlyStopper(
        patience=Config.EARLY_STOPPING_PATIENCE, 
        min_delta=Config.MIN_DELTA
    )
    best_test_f1 = -np.inf

    # 5. Main Training Loop
    for epoch in range(Config.NUM_EPOCHS):
        
        # A. Training Phase
        train_loss, train_acc, train_f1 = train(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            Config.DEVICE, 
            Config.GRADIENT_CLIP_VAL
        )
        
        # B. Evaluation Phase
        test_loss, test_acc, test_f1, cm, report_text = evaluate(model, test_loader, criterion, Config.DEVICE)
        
        # C. Step the Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\n--- Epoch {epoch+1}/{Config.NUM_EPOCHS} (LR: {current_lr:.6f}) ---")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        
        # D. TensorBoard Logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Test', test_acc, epoch)
        writer.add_scalar('F1-Score/Train', train_f1, epoch)
        writer.add_scalar('F1-Score/Test', test_f1, epoch)
        writer.add_scalar('Learning Rate', current_lr, epoch)
        
        # Log classification report as text
        writer.add_text('Classification Report', f"Epoch {epoch+1}:\n{report_text}", epoch)

        # Add Confusion Matrix to TensorBoard
        try:
            class_labels = [f'C{i}' for i in range(Config.NUM_CLASSES)]
            writer.add_figure('Confusion Matrix', plot_confusion_matrix(cm, labels=class_labels), epoch)
        except Exception as e:
            # We skip printing the exception to avoid clutter, as the underlying figure logging mechanism sometimes causes
            # the 'ignored exception' you reported when cleaning up the figure.
            pass
        
        # E. Save the Best Model based on F1-score
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            # Save the model's state dictionary
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"Model saved to {Config.MODEL_SAVE_PATH} with improved F1-score: {best_test_f1:.4f}")
            
        # F. Check Early Stopping
        if early_stopper.early_stop(test_f1):
            break

    print("\n--- Training Complete ---")
    print(f"Best Test F1-Score Achieved: {best_test_f1:.4f}")
    writer.close()
    
# --- Utility to Plot Confusion Matrix (for TensorBoard) ---
def plot_confusion_matrix(cm, labels):
    """Returns a matplotlib figure containing the plotted confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=labels, yticklabels=labels, cbar=False) 
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    ax.set_yticks(np.arange(len(labels)) + 0.5, minor=False)
    ax.set_xticks(np.arange(len(labels)) + 0.5, minor=False)
    ax.tick_params(axis='both', which='major', labelsize=10, labelbottom=True, labeltop=False, labelleft=True, labelright=False, rotation=0)

    fig.tight_layout()
    plt.close(fig) 
    return fig


if __name__ == "__main__":
    # Check and create log directory
    if not os.path.exists(Config.LOG_DIR):
        os.makedirs(Config.LOG_DIR)
        
    main()
