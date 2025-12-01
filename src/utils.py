import torch
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score

# NOTE - Set seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    loop = tqdm(dataloader, desc="Training")
        
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)
        
        loop.set_postfix(loss=loss.item())
        
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    
    return epoch_acc, epoch_loss

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    all_labels = []
    all_preds = []
    all_probs_real = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)
            
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs_real.extend(probs[:, 1].cpu().numpy())
            
    avg_loss = running_loss / total_samples
    avg_acc = correct_predictions / total_samples
    
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    try:
        auc = roc_auc_score(all_labels, all_probs_real)
    except ValueError:
        auc = 0.0
        
    return avg_loss, avg_acc, f1, auc

def save_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    print("Checkpoint saved:", filename)
    
def save_experiment_json(history, filename):
    """@
    Saves the experiment history to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"History saved to {filename}")