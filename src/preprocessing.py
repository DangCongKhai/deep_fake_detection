from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transformations(model_name, use_augmentation):
    """
    Returns the training pipeline using Albumentations.
    
    Resizing depends on the model architecture:
        - simpleCNN      → 256x256
        - xception       → 299x299
        - else           → 224x224  

    Augmentations include:
        - Horizontal flips
        - Shifts, scaling, rotation
        - Noise (GaussNoise or ISONoise)
        - Resize (dynamic)
        - Normalization
        - ToTensorV2

    Args:
        model_name (str): Name of the model architecture.
        use_augmentation (bool): If True, augmentation will be applied.
    Returns:
        A.Compose: Albumentations preprocessing pipeline for training.
    """    
    if model_name.lower() == "simplecnn":
        resize_size = (256, 256)
    elif model_name.lower() == "xception":
        resize_size = (299, 299)
    else:
        resize_size = (224, 224)
        
    # Without augmentation
    if not use_augmentation:
        return A.Compose(
            [
                A.Resize(resize_size[0], resize_size[1]),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2()
            ]
        )
    
    # With augmentation
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=15,
                p= 0.5
            ),
            A.OneOf(
                [
                    A.GaussianBlur(p=0.5),
                    A.ISONoise(
                        color_shift=(0.01, 0.02),
                        intensity=(0.4, 0.5),
                        p=0.5
                    ),
                ],
                p=0.5
            ),
            A.Resize(resize_size[0], resize_size[1]),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ]
    )
    
    
def get_transforms(model_name, split, use_augmentation):
    """
    Master wrapper for selecting the correct transform pipeline.

    Args:
        model_name (str): 'simpleCNN', 'efficientnet_b0', etc.
        split (str): 'train' or 'val' or 'test'
        use_augmentation (bool): If True, augmentation will be applied.
    Returns:
        A.Compose: Albumentations pipeline for the requested split.

    Raises:
        ValueError: If model_name or split is invalid.
    """
    split = split.lower()
    
    if split == 'train':
        return get_transformations(model_name, use_augmentation)
    elif split in ['val', 'valid', 'validation', 'test']:
        return get_transformations(model_name, use_augmentation = False)
    else:
        raise ValueError("Split must be `train` or `val` or `test`")