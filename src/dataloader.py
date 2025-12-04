import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from src.preprocessing import get_transforms

CLASS = ['real', 'fake']

class DeepFakeImageDateset(Dataset):
    """
    Custom dataset for DeepFake images.
    
    Returns both the original image (numpy) and the transformed image (tensor)
    along with the numerical label.
    """
    def __init__(self, data_path, transform=None):
        super().__init__()
        self.data = []
        self.transform = transform
        for label in os.listdir(data_path):
            label_path = os.path.join(data_path, label)
            if not os.path.isdir(label_path):
                continue
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                self.data.append((image_path, label))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path, label = self.data[index]
        
        # Load image
        original_image = Image.open(image_path).convert('RGB')
        image_np = np.array(original_image)
        
        augmented = self.transform(image=image_np)
        transforms_image = augmented['image']
        
        y = torch.tensor(CLASS.index(label), dtype=torch.long)
        
        return transforms_image, y

def get_data_loaders(root_dir: str, model_name:str, batch_size: int = 32):
    """
    Create PyTorch DataLoaders for training and validation image datasets.
    
    This function expects the following directory structure:
        root_dir/
            train/
                <class folders>/
            val/
                <class folders>/
    
    Applies model-specific image transforms and flips labels using `flip_label` so that:
        'real' -> 0
        'fake' -> 1
    
    Args:
        root_dir: Path to the root directory containing 'train' and 'val' subfolders.
        model_name: Name of the model to determine the transforms pipeline
        batch_size: Number of samples per batch. Default is 32.
        
    Returns:
        tuple:
            train_loader (torch.utils.data.DataLoader): DataLoader for training dataset.
            valid_loader (torch.utils.data.DataLoader): DataLoader for validation dataset.
    """
    # Define path
    train_dir = os.path.join(root_dir, 'train')
    valid_dir = os.path.join(root_dir, 'val')
    
    # Transform
    train_transform = get_transforms(model_name, split='train')
    valid_transform = get_transforms(model_name, split='val')
    
    # Dataset
    train_dataset = DeepFakeImageDateset(train_dir, transform=train_transform)
    valid_dataset = DeepFakeImageDateset(valid_dir, transform=valid_transform)
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Class expected: {CLASS}")
    
    # train_classes_found = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    # class_to_idx = {class_name: i for i, class_name in enumerate(train_classes_found)}
    # print(f"Class found: {class_to_idx}")
    actual_labels_found = set([label for _, label in train_dataset.data])
    print(f"Class found: {actual_labels_found}")
    
    return train_loader, valid_loader