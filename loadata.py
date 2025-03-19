import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import (
    Compose, Resize, RandomHorizontalFlip, RandomRotation,
    ColorJitter, GaussianBlur, ToImage, ToDtype, Normalize
)

class GenderDataset(Dataset):
    def __init__(self, root, train=True, transforms=None):
        self.root = root
        self.train = train
        self.transforms = transforms
        self.data = []
        self.labels = []
        self.gender_mapping = ['Female', 'Male']

        phase = 'train' if train else 'test'
        data_dir = os.path.join(root, phase)

        # Duyệt qua các nhóm tuổi
        for age_group in os.listdir(data_dir):
            age_group_path = os.path.join(data_dir, age_group)
            if os.path.isdir(age_group_path):
                # Duyệt qua các thư mục giới tính
                for gender in os.listdir(age_group_path):
                    gender_path = os.path.join(age_group_path, gender)
                    if os.path.isdir(gender_path):
                        for img_name in os.listdir(gender_path):
                            img_path = os.path.join(gender_path, img_name)
                            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.data.append(img_path)
                                self.labels.append(gender)

        if len(self.data) == 0:
            raise ValueError(f"No images found in {data_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        gender = self.labels[idx]

        # Tạo nhãn chỉ dựa trên giới tính
        gender_idx = self.gender_mapping.index(gender)
        label = gender_idx  # Chỉ có 2 lớp: 0 (Female) và 1 (Male)

        # Đọc và xử lý ảnh
        image = cv2.imread(img_path)
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
