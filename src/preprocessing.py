from torchvision import transforms

def get_transforms(model_name='simpleCNN', split='train'):
    """
    Returns transforms based on model requirements.
        - simpleCNN: expects 256x256 and simple 0-1 scaling
        - efficientnet: expects 224x224 and ImageNet normalization
    """
    if model_name.lower() == 'simplecnn':
        transform_list = [
            transforms.Resize((256, 256)),
            transforms.ToTensor()]
        
    elif 'efficientnet' in model_name.lower():
        transform_list = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])]
        
    else: 
        raise ValueError(f"Unknown model name: {model_name}")
    
    return transforms.Compose(transforms=transform_list)