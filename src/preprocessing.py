from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# def get_transforms(model_name='simpleCNN', split='train'):
#     """
#     Returns transforms based on model requirements.
#         - simpleCNN: expects 256x256 and simple 0-1 scaling
#         - efficientnet: expects 224x224 and ImageNet normalization
#     """
#     if model_name.lower() == 'simplecnn':
#         transform_list = [
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], 
#                 std=[0.229, 0.224, 0.225])]
        
#     elif 'efficientnet' in model_name.lower():
#         transform_list = [
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                  std=[0.229, 0.224, 0.225])]
        
#     else: 
#         raise ValueError(f"Unknown model name: {model_name}")
    
#     return transforms.Compose(transforms=transform_list)

def get_train_transforms(model_name: str = "simpleCNN"):
    """
    Returns the training augmentation pipeline using Albumentations.
    
    Resizing depends on the model architecture:
        - simpleCNN      → 256x256
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

    Returns:
        A.Compose: Albumentations preprocessing pipeline for training.
    """    
    if model_name.lower() == "simplecnn":
        resize_size = (256, 256)
    elif model_name.lower() == "xception":
        resize_size = (299, 299)
    else:
        resize_size = (224, 224)
        
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
    
def get_val_transforms(model_name='simpleCNN'):
    """
    Returns validation/test preprocessing pipeline.

    Resizing depends on the model architecture:
        - simpleCNN      → 256x256
        - xception       → 299x299
        - else           → 224x224

    Validation set must NOT apply noise or random augmentations.

    Args:
        model_name (str): Name of the model architecture.

    Returns:
        A.Compose: Albumentations preprocessing pipeline for validation/testing.
    """
    if model_name.lower() == 'simplecnn':
        resize_size = (256, 256)
    elif model_name.lower() == "xception":
        resize_size = (299, 299)
    else:
        resize_size = (224, 224)
        
    return A.Compose(
        [
            A.Resize(resize_size[0], resize_size[1]),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ]
    )
    
def get_transforms(model_name='simpleCNN', split='train'):
    """
    Master wrapper for selecting the correct transform pipeline.

    Args:
        model_name (str): 'simpleCNN', 'efficientnet_b0', etc.
        split (str): 'train' or 'val'

    Returns:
        A.Compose: Albumentations pipeline for the requested split.

    Raises:
        ValueError: If model_name or split is invalid.
    """
    split = split.lower()
    
    if split == 'train':
        return get_train_transforms(model_name)
    elif split in ['val', 'valid', 'validation', 'test']:
        return get_val_transforms(model_name)
    else:
        raise ValueError("Split must be `train` or `val` or `test`")