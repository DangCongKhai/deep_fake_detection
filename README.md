# **DeepFake Detection System**

This repository provides a comprehensive Deepfake Detection System built using Deep Learning. It covers the **full pipeline**, from dataset preprocessing and model training to deployment via a Streamlit Web Application.

## **Key Features**
- Deepfake detection using CNN-based architectures.
- Face preprocessing and transformation pipeline
- Fine-tuned models with augmentation
- Streamlit web interface for real-time inference
- Clean project structure with reproducible dependency management using `uv`.

## **Project Architectures**
```bash
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

