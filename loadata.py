import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import (
    Compose, Resize, RandomHorizontalFlip, RandomRotation,
    ColorJitter, GaussianBlur, ToImage, ToDtype, Normalize
)

class AgeGenderDataset(Dataset):
    def __init__(self, root, train=True, transforms=None):
        self.root = root
        self.train = train
        self.transforms = transforms
        self.data = []
        self.labels = []
        self.age_mapping = ['16-20', '21-25', '26-30', '31-35', '36-40',
                            '41-45', '46-50', '51-55', '56-60', '61-70']
        self.gender_mapping = ['Female', 'Male']

        phase = 'train' if train else 'test'
        data_dir = os.path.join(root, phase)

        for age_group in os.listdir(data_dir):
            age_group_path = os.path.join(data_dir, age_group)
            if os.path.isdir(age_group_path):
                for gender in os.listdir(age_group_path):
                    gender_path = os.path.join(age_group_path, gender)
                    if os.path.isdir(gender_path):
                        for img_name in os.listdir(gender_path):
                            img_path = os.path.join(gender_path, img_name)
                            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.data.append(img_path)
                                self.labels.append((age_group, gender))

        if len(self.data) == 0:
            raise ValueError(f"No images found in {data_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        age_group, gender = self.labels[idx]

        # Tạo nhãn
        age_idx = self.age_mapping.index(age_group)
        gender_idx = self.gender_mapping.index(gender)
        label = age_idx * 2 + gender_idx  # 10 tuổi * 2 giới tính = 20 lớp

        # Đọc và xử lý ảnh
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image = self.transforms(image)

        return image, label

# Transform cho tập train
train_transform = Compose([
    ToImage(),
    Resize((224, 224)),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(15),
    ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ToDtype(torch.float32, scale=True),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transform cho tập validation/test và dự đoán
val_transform = Compose([
    ToImage(),
    Resize((224, 224)),
    ToDtype(torch.float32, scale=True),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])