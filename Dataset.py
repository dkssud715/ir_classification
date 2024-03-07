import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.io import read_image
import torchvision.transforms as transforms
import pandas as pd

class GasVidDataset(Dataset):
    def __init__(self, data_dir, labeling_file, list_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(labeling_file, names=['file_name','label'])
        self.img_lists = pd.read_csv(list_file)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image_path = self.img_labels.iloc[index]
        if os.path.exists(image_path):
            image = read_image(image_path)
            label = self.img_labels.iloc[index,1]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label

    def __len__(self):
        return len(self.img_labels)