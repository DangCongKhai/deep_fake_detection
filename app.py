import os
import cv2
import tempfile
import numpy as np
from PIL import Image
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
from src.model import SimpleCNN, EfficientNetDF, XceptionDF, Resnet18DF

import torch
import torch.nn as nn
from facenet_pytorch import MTCNN
from torchvision import transforms, models
from utils import GradCAM, FaceCropper, get_preprocessing, denormalize, get_device, get_last_conv_layer


PROJECT_ROOT = Path(os.getcwd()).resolve()
MODEL_DIR = PROJECT_ROOT / "model_checkpoints"

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


@st.cache_resource
def load_model(model_name, model_path, device):
    model_name_lower = model_name.lower()
    if model_name_lower == 'simplecnn':
        model = SimpleCNN(num_classes=2).to(device)
        target_layer = model.conv4
    elif model_name_lower == 'efficientnet':
        model = EfficientNetDF(num_classes=2, pretrained=False).to(device)
        target_layer = get_last_conv_layer(model)
    elif model_name_lower == 'xception':
        model = XceptionDF(num_classes=2)
        target_layer = get_last_conv_layer(model)
    elif model_name_lower == 'resnet':
        model = Resnet18DF(num_classes=2)
        target_layer = get_last_conv_layer(model)
    else:
        return None, None
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(sd)

    except Exception as e:
        st.write(f"Error loading model: {e}")
        return None, None
    
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True

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

# MAIN APP
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
    "SimpleCNN": MODEL_DIR / "simplecnn.pth",
    "EfficientNet": MODEL_DIR / "efficientnet.pth",
    "Xception": MODEL_DIR / "xception.pth",
    "ResNet": MODEL_DIR / "resnet.pth"}


CURRENT_MODEL_PATH = MODEL_PATHS[MODEL_CHOICE]

model, target_layer = load_model(MODEL_CHOICE, CURRENT_MODEL_PATH, DEVICE)

# === MAIN CONTENT ===
uploaded_file = st.file_uploader("Choose a file...", type=['jpg', 'png', 'jpeg', 'mp4', 'avi', 'gif'])

if uploaded_file is not None:
    # Display uploaded content
    st.divider()
    
    column1, column2 = st.columns([1, 3])
    with column1:
        st.subheader("1. Input Preview")
        
        fname = uploaded_file.name
        if fname.lower().endswith(('.mp4', '.avi', '.mov')):
            st.video(uploaded_file, width=300)
        else:
            st.image(uploaded_file, width=300) # Simple fixed width for preview
            
        # Reset pointer
        uploaded_file.seek(0)
        _, col_center, _ = st.columns([1, 4, 1])
        if col_center.button("🔍 Analyze Content"): # Make this center aligned
            with st.spinner(f"Loading {MODEL_CHOICE} and processing frames..."): # Make this center aligned
                
                # 2. Process File
                data_pairs = process_file(uploaded_file)
                if not data_pairs:
                    st.stop() # Make this center aligned
                    
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
                
                # 5. Results calculation
                avg_fake_prob = np.mean(probs)
                prediction = "FAKE" if avg_fake_prob > 0.5 else "REAL"
                confidence = avg_fake_prob if prediction == 'FAKE' else 1 - avg_fake_prob
            
            # === DISPLAY RESULTS ===
            with column2:
              
                st.subheader("2. Analysis Result")
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
                    # How can we make the plot larger?
                    
                    if vis_data:
                        full_orig, face_crop, cam_overlay = vis_data
                        
                        fig, axs = plt.subplots(1, 3, figsize=(17, 12))
                        
                        # 1. Original
                        axs[0].imshow(full_orig)
                        axs[0].set_title("Original Frame", fontsize = 25)
                        axs[0].axis('off')
                        
                        # 2. Face Crop
                        axs[1].imshow(face_crop)
                        axs[1].set_title("Cropped Face", fontsize = 25)
                        axs[1].axis('off')
                        
                        # 3. Grad-CAM
                        axs[2].imshow(cam_overlay)
                        axs[2].set_title(f"Grad-CAM ({prediction})", fontsize = 25)
                        axs[2].axis('off')
                        
                        st.pyplot(fig)
else:
    st.info("Please upload an image or video to start.")