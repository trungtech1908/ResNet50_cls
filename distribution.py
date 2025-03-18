import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def count_images(root):
    distribution = {'train': defaultdict(lambda: {'Female': 0, 'Male': 0}),
                    'test': defaultdict(lambda: {'Female': 0, 'Male': 0})}

    for phase in ['train', 'test']:
        data_dir = os.path.join(root, phase)
        if not os.path.exists(data_dir):
            continue

        for age_group in os.listdir(data_dir):
            age_group_path = os.path.join(data_dir, age_group)
            if os.path.isdir(age_group_path):
                for gender in os.listdir(age_group_path):
                    gender_path = os.path.join(age_group_path, gender)
                    if os.path.isdir(gender_path):
                        num_images = len(
                            [f for f in os.listdir(gender_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                        distribution[phase][age_group][gender] += num_images

    return distribution


def plot_distribution(distribution):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    phases = ['train', 'test']
    colors = {'Female': 'lightcoral', 'Male': 'royalblue'}

    for i, phase in enumerate(phases):
        age_groups = sorted(distribution[phase].keys())
        females = [distribution[phase][age]['Female'] for age in age_groups]
        males = [distribution[phase][age]['Male'] for age in age_groups]

        x = range(len(age_groups))
        width = 0.4

        axes[i].bar(x, females, width, label='Female', color=colors['Female'])
        axes[i].bar([p + width for p in x], males, width, label='Male', color=colors['Male'])

        axes[i].set_xlabel('Age Groups')
        axes[i].set_ylabel('Number of Images')
        axes[i].set_title(f'Dataset Distribution: {phase.capitalize()}')
        axes[i].set_xticks([p + width / 2 for p in x])
        axes[i].set_xticklabels(age_groups, rotation=45)
        axes[i].legend()
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


# Thư mục gốc chứa dữ liệu
data_root = r"D:\AgeGender_Classification\Data"  # Cập nhật đường dẫn dữ liệu của bạn

distribution = count_images(data_root)
plot_distribution(distribution)
