import torch
import numpy as np
import cv2
from facenet_pytorch import MTCNN
from torchvision import transforms
import torch.nn as nn
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # Robust hooking strategy
        self.handle = target_layer.register_forward_hook(self.save_activation_and_hook)
        
    def save_activation_and_hook(self, module, input, output):
        self.activations = output
        output.register_hook(self.save_gradient)
        
    def save_gradient(self, grad):
        self.gradients = grad
        
    def __call__(self, x, class_idx=None):
        self.gradients = None
        self.activations = None
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
        self.model.zero_grad()
        target = output[:, class_idx]
        target.backward(retain_graph=True)
        
        if self.gradients is None:
            return np.zeros((x.shape[2], x.shape[3])), 0.0
        
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.shape[2], x.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        cam[cam < 0.2] = 0 
        return cam, torch.softmax(output, dim=1)[0][1].item()
    
    def remove_hooks(self):
        self.handle.remove()
        
class FaceCropper:
    def __init__(self, device='cpu'):
        self.mtcnn = MTCNN(
            keep_all=False, 
            select_largest=True, 
            device=device, 
            margin=20)

    def crop(self, pil_image):
        try:
            face_tensor = self.mtcnn(pil_image)
            
            if face_tensor is not None:
                boxes, _ = self.mtcnn.detect(pil_image)
                if boxes is not None:
                    box = boxes[0]
                    x1, y1, x2, y2 = [int(b) for b in box]
                    width, height = pil_image.size
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(width, x2); y2 = min(height, y2)
                    
                    return pil_image.crop((x1, y1, x2, y2))
            return None
        except Exception as e:
            return None
        
def get_preprocessing(model_name):
    model_name = model_name.lower()
    if model_name in ['simplecnn', 'simple_cnn']: 
        IMG_SIZE = (256, 256)
    elif model_name == "xception":
        IMG_SIZE = (299, 299)
    else:
        IMG_SIZE = (224, 224)
        
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
             std=[0.229, 0.224, 0.225]
        )
    ])
    
def denormalize(tensor):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std])
    img = inv_normalize(tensor)
    img = img.permute(1, 2, 0).cpu().numpy()
    return np.clip(img, 0, 1)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"
    
    
def get_last_conv_layer(model):
    last_conv_layer = None
    # Iterate through all modules in the model
    for module in model.modules():
        # Check if the module is a convolutional layer (Conv2d, Conv3d, etc.)
        if isinstance(
            module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)
        ):
            last_conv_layer = module
    return last_conv_layer