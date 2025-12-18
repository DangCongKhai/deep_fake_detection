import os
import cv2
import tempfile
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

import timm
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN
from torchvision import transforms, models

st.set_page_config(
    page_title='Deepfake Detection',
    page_icon="🕵️‍♂️",
    layout='wide')

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            background-color: #ff4b4b;
            color: white;
            font-weight: bold;
        }
        h1 {
            color: #1f1f1f;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# MODEL DEFINITIONS
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.flatten_dim = 16 * 16 * 16
        self.fc1 = nn.Linear(self.flatten_dim, 16)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class EfficientNetDF(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(EfficientNetDF, self).__init__()
        self.model = models.efficientnet_b0(weights=None) 
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(num_features, num_classes)
        )
    def forward(self, x):
        return self.model(x)
    
class Resnet18DF(nn.Module):
    def __init__(self, num_classes = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        num_ftrs = self.model.fc.in_features
        params = list(self.model.parameters())
        for param in params:
            param.requires_grad = False
            
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, X):
        return self.model(X)
    
class XceptionDF(nn.Module):
    def __init__(self, num_classes = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = timm.create_model("xception", pretrained=True)
        num_ftrs = self.model.fc.in_features
        for param in self.model.parameters():
            param.requires_grad = False # Freeze base model
            
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, X):
        return self.model(X)
    
# ==========================================
# UTILS (GradCAM & Cropper)
# ==========================================
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
            st.error(f"Face detection error: {e}")
            return None
        
# ==========================================
# HELPER FUNCTIONS
# ==========================================
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

@st.cache_resource
def load_model(model_name, model_path, device):
    model_name_lower = model_name.lower()
    if model_name_lower == 'simplecnn':
        model = SimpleCNN(num_classes=2)
        target_layer = model.conv4
    elif model_name_lower == 'efficientnet':
        model = EfficientNetDF(num_classes=2, pretrained=False)
        target_layer = model.model.features[-1]
    elif model_name_lower == 'xception':
        model = XceptionDF(num_classes=2)
        target_layer = model.model.conv4
    elif model_name_lower == 'resnet':
        model = Resnet18DF(num_classes=2)
        target_layer = model.model.layer4[-1]
    else:
        return None, None
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(sd)
    except Exception as e:
        st.error(f"Failed to load weights: {e}")
        return None, None
    
    model.to(device).eval()
    return model, target_layer

def process_file(file, num_frames=10):
    cropper = FaceCropper(device='cpu')
    
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())
    file_path = tfile.name
    
    if hasattr(file, 'name'):
        ext = os.path.splitext(file.name)[1].lower()
    else:
        ext = '.png'
        
    raw_frames = []
    
    try:
        if ext in ['.jpg', '.png', '.jpeg']:
            raw_frames = [Image.open(file_path).convert('RGB')]
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif']:
            cap = cv2.VideoCapture(file_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 0:
                indices = np.linspace(0, total-1, num_frames, dtype=int)
                for i in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret:
                        raw_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cap.release()
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return []
    
    final_data = []
    progress_bar = st.progress(0)
    for i, frame in enumerate(raw_frames):
        face = cropper.crop(frame)
        if face is not None:
            final_data.append((frame, face))
        progress_bar.progress((i + 1) / len(raw_frames))
        
    progress_bar.empty()
    
    if not final_data:
        st.warning("No faces detected! Using full frames")
        return [(f, f) for f in raw_frames]
    
    return final_data

# ==========================================
# MAIN APP
# ==========================================
st.title("🕵️‍♂️ Deepfake Detection Lab")
st.write("Upload an image or video to detect if it's **Real** or **Fake**.")

# === SIDEBAR CONFIG ===
st.sidebar.header("Configuration")

# Device Selection
DEVICE = torch.device(get_device())
st.sidebar.write(f"**Device**: `{DEVICE}`")

# Model Selection
MODEL_CHOICE = st.sidebar.radio(
    "Choose Model Architecture:",
    ('SimpleCNN', "EfficientNet", "Xception", "ResNet"))

# Update your paths here
MODEL_PATHS = {
    "SimpleCNN": "D:/VNUK ASSIGNMENTS/3rd YEAR/1st SEM/AI/deepfake-detection/deep_fake_detection/model_checkpoints/simplecnn.pth",
    "EfficientNet": "D:/VNUK ASSIGNMENTS/3rd YEAR/1st SEM/AI/deepfake-detection/deep_fake_detection/model_checkpoints/efficientnet.pth",
    "Xception": "D:/VNUK ASSIGNMENTS/3rd YEAR/1st SEM/AI/deepfake-detection/deep_fake_detection/model_checkpoints/xception.pth",
    "ResNet": "D:/VNUK ASSIGNMENTS/3rd YEAR/1st SEM/AI/deepfake-detection/deep_fake_detection/model_checkpoints/resnet.pth"}

CURRENT_MODEL_PATH = MODEL_PATHS[MODEL_CHOICE]

# === MAIN CONTENT ===
uploaded_file = st.file_uploader("Choose a file...", type=['jpg', 'png', 'jpeg', 'mp4', 'avi', 'gif'])

if uploaded_file is not None:
    # Display uploaded content
    st.divider()
    st.subheader("1. Input Preview")
    
    fname = uploaded_file.name
    if fname.lower().endswith(('.mp4', '.avi', '.mov')):
        st.video(uploaded_file)
    else:
        st.image(uploaded_file, width=500) # Simple fixed width for preview
        
    # Reset pointer
    uploaded_file.seek(0)
    
    if st.button("🔍 Analyze Content"):
        with st.spinner(f"Loading {MODEL_CHOICE} and processing frames..."):
            # 1. Load model
            model, target_layer = load_model(MODEL_CHOICE, CURRENT_MODEL_PATH, DEVICE)
            if model is None:
                st.stop()
            
            # 2. Process File
            data_pairs = process_file(uploaded_file)
            if not data_pairs:
                st.stop()
                
            # 3. Init Tools
            grad_cam = GradCAM(model, target_layer)
            preprocess = get_preprocessing(MODEL_CHOICE)
            
            probs = []
            vis_data = None
            vis_idx = len(data_pairs) // 2
            
            # 4. Inference Loop
            progress_text = "Running Inference..."
            my_bar = st.progress(0, text=progress_text)
            
            for i, (orig_frame, cropped_frame) in enumerate(data_pairs):
                # Prepare input
                input_tensor = preprocess(cropped_frame).unsqueeze(0).to(DEVICE).requires_grad_(True)
                
                # Grad-CAM
                heatmap, prob_fake = grad_cam(input_tensor, class_idx=1)
                probs.append(prob_fake)
                
                # Save visualization for middle frame
                if i == vis_idx:
                    denorm_crop = denormalize(input_tensor[0].detach().cpu())
                    hmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                    hmap_color = cv2.cvtColor(hmap_color, cv2.COLOR_BGR2RGB)
                    hmap_color = np.float32(hmap_color) / 255
                    overlay = 0.6 * denorm_crop + 0.4 * hmap_color
                    vis_data = (orig_frame, denorm_crop, np.clip(overlay, 0, 1))
                    
                my_bar.progress((i + 1) / len(data_pairs), text=progress_text)
                
            my_bar.empty()
            grad_cam.remove_hooks()
            
            # 5. Results calcultation
            avg_fake_prob = np.mean(probs)
            prediction = "FAKE" if avg_fake_prob > 0.5 else "REAL"
            confidence = avg_fake_prob if prediction == 'FAKE' else 1 - avg_fake_prob
            
            # === DISPLAY RESULTS ===
            st.divider()
            st.subheader("2. Analysis Results")
            
            # Metrics Column
            col1, col2 = st.columns([1, 2])
            with col1:
                if prediction == 'FAKE':
                    st.error(f"### {prediction}")
                else:
                    st.success(f"### {prediction}")
                    
                st.metric("Confidence Score", f"{confidence * 100:.2f}%")
                st.metric("Fake Probability", f"{avg_fake_prob:.4f}")
                
            with col2:
                # Plotting using Matplotlib
                if vis_data:
                    full_orig, face_crop, cam_overlay = vis_data
                    
                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # 1. Original
                    axs[0].imshow(full_orig)
                    axs[0].set_title("Original Frame")
                    axs[0].axis('off')
                    
                    # 2. Face Crop
                    axs[1].imshow(face_crop)
                    axs[1].set_title("Face Input")
                    axs[1].axis('off')
                    
                    # 3. Grad-CAM
                    axs[2].imshow(cam_overlay)
                    axs[2].set_title(f"Grad-CAM ({prediction})")
                    axs[2].axis('off')
                    
                    st.pyplot(fig)
else:
    st.info("Please upload an image or video to start.")