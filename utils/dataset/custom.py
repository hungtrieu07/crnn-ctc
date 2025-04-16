import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from utils.logger import LOGGER

class CustomPlateDataset(Dataset):
    def __init__(self, data_root, label_file, input_shape=(168, 48), is_train=True):
        self.data_root = data_root
        self.input_shape = input_shape
        self.is_train = is_train
        self.image_paths = []
        self.labels = []

        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                image_name, label = line.strip().split('\t')
                cleaned_label = ''.join(label.split())
                if not cleaned_label:
                    LOGGER.warning(f"Empty label for image: {image_name}")
                    continue
                self.image_paths.append(os.path.join(data_root, image_name))
                self.labels.append(cleaned_label)

        # Define transforms
        common_transforms = [
            transforms.Resize((input_shape[1], input_shape[0])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomRotation(10),  # Rotate ±10 degrees
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random shift
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness/contrast
            ] + common_transforms)
        else:
            self.transform = transforms.Compose(common_transforms)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        image = self.transform(image)
        return image, label

    def convert(self, targets):
        from utils.converter import StrLabelConverter
        converter = StrLabelConverter()
        converted_targets = []
        for target in targets:
            try:
                indices, _ = converter.encode(target)
                converted_targets.append(indices)
            except KeyError as e:
                LOGGER.error(f"Invalid character in label: {target}, error: {e}")
                raise
        return converted_targets