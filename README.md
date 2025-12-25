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

<!-- ## Create a Pull Request (PR)

Follow these steps to create a clear, reviewable pull request:

1. Update your local `develop` branch:

```
git checkout develop
git pull origin develop
```

2. Create a new feature branch (use a descriptive name):

```
git checkout -b feature/short-description
```

Follow git convention here: [Link](https://gist.github.com/qoomon/5dfcdf8eec66a051ecd85625518cfd13)

3. Make your changes, run tests/lint locally, then commit:

```
git add .
git commit -m "Short: describe what and why"
```

4. Push the branch to the remote and open a PR:

```
git push -u origin feature/short-description
# Then open a PR on GitHub from your branch into `main`
```

You can also create a PR from the command line with [GitHub CLI](https://cli.github.com):

```
gh pr create --base main --head your-branch --title "Short title" --body "Detailed description"
```

PR checklist (suggested):
- Include a descriptive title and detailed description
- Link related issue(s) if present
- Request reviewers and set the correct target branch -->

