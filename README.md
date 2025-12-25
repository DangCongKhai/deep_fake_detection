# **DeepFake Detection System**
![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNNs-FEB05D.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b.svg)
![License](https://img.shields.io/badge/License-Academic-8a2be2.svg)
![UV](https://img.shields.io/badge/Dependency%20Manager-uv-4B9DA9.svg)
---

This repository provides a comprehensive Deepfake Detection System built using Deep Learning. It covers the **full pipeline**, from dataset preprocessing and model training to deployment via a Streamlit Web Application.

## **Key Features**
- Deepfake detection using CNN-based architectures.
- Face preprocessing and transformation pipeline
- Fine-tuned models with augmentation
- Streamlit web interface for real-time inference
- Clean project structure with reproducible dependency management using `uv`.

## **Project Architectures**
```python
deep_fake_detection/
│
├── .github/                    # GitHub workflows and configuration
├── .venv/                      # Local virtual environment (ignored)
│
├── data/                       # Data for the project
│
├── model_checkpoints/          # Saved model weights and checkpoints
│
├── notebooks/                  # Research & experimentation notebooks
│   ├── efficient-net.ipynb
│   ├── face-crop-grad-cam.ipynb
│   ├── fourier_transform.ipynb
│   ├── resnet.ipynb
│   ├── simple-cnn.ipynb
│   └── xception.ipynb
│
├── results/                    # Model evaluation outputs
│   ├── efficient-net/
│   ├── resnet18/
│   ├── simple-cnn/
│   └── xception/
│
├── src/                        # Core DeepFake detection pipeline
│   ├── __init__.py
│   ├── dataloader.py           # Dataset loading & batching
│   ├── model.py                # CNN architectures
│   ├── preprocessing.py        # Face detection & image transforms
│   └── utils.py                # Helper functions
│
├── app.py                      # Streamlit web application entry point
├── dataset_construction.ipynb  # Dataset preparation & labeling
├── pyproject.toml              # Project dependencies (uv)
├── uv.lock                     # Locked dependency versions
├── .python-version             # Python version pin
├── .gitignore
├── utils.py                    # Shared utility helpers
└── README.md
```

## **Project Setup**
### **Clone the Repo and Setup**
1. Clone the repository: `git clone https://github.com/DangCongKhai/deep_fake_detection.git`
2. Dependencies: 
- Install uv from [here](https://docs.astral.sh/uv/getting-started/installation/)
- Sync dependencies: In your terminal, run

  ```bash
  uv sync
  ```
- Add new packages or dependencies: run 
  
  ```bash
  uv add [package-name]
  ```

### **Dataset Setup**
This project uses [**FaceForensics++ extracted frames**](https://www.kaggle.com/datasets/adham7elmy/faceforencispp-extracted-frames) dataset, downloaded using *KaggleHub*:
#### 1. Install KaggleHub
If not already installed:
```bash
uv add kagglehub
```

#### 2. Download the Dataset
The dataset is downloaded programmatically using KaggleHub:
```python
import kagglehub

dataset_path = kagglehub.dataset_download(
    "adham7elmy/faceforencispp-extracted-frames"
)

print("Dataset downloaded to:", dataset_path)
```
The dataset will be cached locally by KaggleHub.

#### Construct the Final Dataset
After downloading, run the dataset construction notebook [`dataset_construction.ipynb`](dataset_construction.ipynb).

This notebook:
- Organizes real vs fake samples.
- Applies preprocessing and labelling.
- Prepares the dataset structure used by the training pipeline

**Important**: Make sure the generated dataset is placed inside the `data/` directory before training.

#### Expected Dataset Structure
After running the notebook, your `data/` directory should resemble:

```python
data/
├── train/
│   ├── fake/
│   └── real/
├── test/
│   ├── fake/
│   └── real/
├── val/
│   ├── fake/
│   └── real/
```

## **Model Performance**
We evaluate multiple CNN architectures. All training histories and confusion matrices are available in the [`results/`](results/) directory.

|Model|Status|Example Output|
|---|---|---|
|EfficientNet|Fine-tuned + Augmentation|[`efficientnet_with_aug_finetuned.png`](results/efficient-net/efficientnet_with_aug_finetuned.png)|
|Xception|Fine-tuned + Augmentation|[`Xception_augmentation_finetuned_full_model_deepfake_detection.png`](results/xception/Xception_augmentation_finetuned_full_model_deepfake_detection.png)|
|ResNet|Fine-tuned + Augmentation|[`Resnet18_augmentation_finetuned_deepfake_detection.png`](results/resnet18/Resnet18_augmentation_finetuned_deepfake_detection.png)|

## **Run the Web Application**
The project includes a **Streamlit web app** for interactive Deepfake detection. To run the web app, run the following command:

```bash
streamlit run app.py
```
Upload an image or video frame to get a real-time prediction.

## **Contributors**
This project is collaboratively developed by:
- **Dang Cong Khai**
  - https://github.com/DangCongKhai
- **Hoang Mai Duc Kien**
  - https://github.com/ndrhmdk