import torch
import cv2
from torch.utils.data import Dataset

import os
class DeepFakeImageDataset(Dataset):
    def __init__(self, data_path, classes, transform):
        super().__init__()
        self.data = []
        self.classes = classes
        self.transform = transform
        for label in os.listdir(data_path):
            label_path = os.path.join(data_path, label)
            for image in os.listdir(label_path):
                image_path = os.path.join(label_path, image)
                self.data.append((image_path, label))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        try:
            transformed_image = self.transform(image=original_image)["image"]
        except:
            transformed_image = self.transform(original_image)

        y = self.classes.index(label)
        return original_image, transformed_image, torch.tensor(y)
