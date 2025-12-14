import torch
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from .train import get_all_predictions
plt.style.use('ggplot')

# NOTE - Set seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

# NOTE. Save Checkpoint & Experiment
def save_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    print("Checkpoint saved:", filename)
    
def save_experiment_json(history, filename):
    """
    Saves the experiment history to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"History saved to {filename}")
 

# NOTE. Get device   
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"
    

def performance(model, loader, device, model_name, task_name, result_path, class_names=['real', 'fake']):
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    _, y_true, y_pred = get_all_predictions(model, loader, device)
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy score: {acc:.4f}")
    
    print("Classification Report")
    print(classification_report(y_true, y_pred))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='crest',
        xticklabels=class_names,
        yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{model_name} Confusion Matrix')
    img_save = f"{result_path}/{model_name}_{task_name}_cfm.png"
    plt.savefig(img_save)
    plt.show()