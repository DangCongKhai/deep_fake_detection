import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from PIL import Image
from .preprocessing import get_transformations
from .train import get_all_predictions
import os
import re

plt.style.use("ggplot")

current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, ".."))


# NOTE - Set seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# NOTE. Save Checkpoint & Experiment
def save_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(checkpoint, filename)
    print("Checkpoint saved:", filename)


def save_experiment_json(history, filename):
    """
    Saves the experiment history to a JSON file.
    """
    with open(filename, "w") as f:
        json.dump(history, f, indent=4)
    print(f"History saved to {filename}")


# NOTE. Get device
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


# NOTE. Plot History
def plot_history(history, title, save_path=None, start_finetuned_epoch=None):
    epochs = range(1, len(history["train_losses"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    plt.suptitle(title)

    ax[0].plot(
        epochs,
        history["train_losses"],
        label="training loss",
        marker="*",
        color="green",
    )
    ax[0].plot(
        epochs, history["val_losses"], label="validation loss", marker="s", color="blue"
    )
    ax[0].set_title("Loss Curve")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    if start_finetuned_epoch is not None:
        ax[0].axvline(
            x=start_finetuned_epoch,
            color="red",
            linestyle="--",
            label="Start Fine-Tuning",
        )
    ax[0].legend()
    ax[0].grid(True, linestyle="--", alpha=0.7)

    ax[1].plot(
        epochs,
        history["train_accuracy"],
        label="training accuracy",
        marker="*",
        color="green",
    )
    ax[1].plot(
        epochs,
        history["val_accuracy"],
        label="validation accuracy",
        marker="s",
        color="blue",
    )
    ax[1].set_title("Accuracy Curve")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    if start_finetuned_epoch is not None:
        ax[1].axvline(
            x=start_finetuned_epoch,
            color="red",
            linestyle="--",
            label="Start Fine-Tuning",
        )
    ax[1].grid(True, linestyle="--", alpha=0.7)

    if save_path is not None:
        plt.savefig(save_path)
    plt.tight_layout()
    plt.show()


# NOTE. Performance Evaluation
def performance(
    model, loader, device, model_name, save_path=None, class_names=["real", "fake"]
):
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

    _, y_true, y_pred = get_all_predictions(model, loader, device)

    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy score: {acc:.4f}")

    print("Classification Report")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="crest",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{model_name} Confusion Matrix")
    if save_path is not None:
        plt.savefig(save_path)
    plt.tight_layout()
    plt.show()


def get_labels(logits):
    return (
        torch.argmax(nn.Softmax(dim=-1)(logits), dim=1)
        .cpu()
        .detach()
        .flatten()
        .tolist()
    )


# NOTE. Error analysis
def error_analysis(
    model,
    fake_image_path,
    model_name,
    device,
    batch_size=32,
    use_fourier_transform=False,
):
    fake_methods = [
        "Deepfakes",
        "Face2Face",
        "FaceSwap",
        "FaceShifter",
        "NeuralTextures",
    ]
    summary = {}
    for method in fake_methods:
        summary[method] = {"total": 0, "correct_pred": 0}
    items = os.listdir(fake_image_path)
    model.eval()
    transform = get_transformations(model_name, use_augmentation=False)
    pattern = r"(train|val|test)"
    data_split = re.findall(pattern, fake_image_path)[0]
    for i in range(0, len(items), batch_size):
        items_batch = items[i : i + batch_size]
        images_batch = []
        fourier_mags_batch = []
        methods = []
        # Read each image and stack
        for item in items_batch:
            method = item.split("_")[2]  # Eg: fake_849_Deepfakes_051.png -> Deepfakes
            methods.append(method)
            summary[method]["total"] += 1
            image_path = os.path.join(fake_image_path, item)
            image_name = item.split(".")[0]

            image = np.array(Image.open(image_path).convert("RGB"))
            image = transform(image=image)["image"]
            images_batch.append(image)

            if use_fourier_transform:
                # Get fourier transform
                fourier_mag = torch.tensor(
                    np.load(f"{project_root}/fourier/{data_split}/{image_name}.npy"),
                    dtype=torch.float32,
                ).unsqueeze(0)
                fourier_mags_batch.append(fourier_mag)
        batch_tensors = torch.stack(images_batch, dim=0).to(device=device)
        
        
        with torch.inference_mode():
            if use_fourier_transform:
                batch_fourier_mags_tensor = torch.stack(fourier_mags_batch, dim=0).to(device=device)
                logits = model(batch_fourier_mags_tensor, batch_tensors)
            else:
                logits = model(batch_tensors)
            labels = get_labels(logits)
            for label, method in zip(labels, methods):
                if label == 1:
                    summary[method]["correct_pred"] += 1
    for method in fake_methods:
        print(
            f"{method} : {summary[method]['correct_pred'] / summary[method]['total']:.4f}"
        )
    return summary


def plot_error_analysis(summary, model_name, save_path=None):
    methods = list(summary.keys())
    values = [round(summary[method]['correct_pred']/summary[method]['total'], 4) for method in methods]
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#CC66FF'] 
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(methods, values, color=colors)
    ax.bar_label(bars)
    plt.xlabel('Facial Forgery Methods')
    plt.ylabel('Accuracy')
    plt.xlim(0, 1.1)
    plt.title(f'Accuracy on different facial forgery methods for {model_name}')
    plt.grid(False)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()