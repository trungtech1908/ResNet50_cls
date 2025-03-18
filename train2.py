import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import shutil
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
import argparse
import matplotlib.pyplot as plt
import numpy as np
from loadata import AgeGenderDataset, train_transform, val_transform


def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(30, 30))
    plt.imshow(cm, interpolation='nearest', cmap="Wistia")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


def get_args():
    parser = argparse.ArgumentParser("Train Arguments")
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--num_epoch', '-nep', type=int, default=1000)
    parser.add_argument('--num_workers', '-n', type=int, default=2)
    parser.add_argument('--log_path', '-lp', type=str, default='Record')
    parser.add_argument('--root', '-r', type=str,
                        default=r"D:\AgeGender_Classification\Data",
                        help='Path to dataset root directory')
    parser.add_argument('--checkpoint_path', '-cpp', type=str, default='checkpoint')
    parser.add_argument('--prepare_checkpoint_path', '-pre', type=str, default=None)
    return parser.parse_args()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)
    writer = SummaryWriter(args.log_path)

    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    train_dataset = AgeGenderDataset(root=args.root,
                                     train=True,
                                     transforms=train_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  drop_last=True)

    val_dataset = AgeGenderDataset(root=args.root,
                                   train=False,
                                   transforms=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                drop_last=False)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    model.fc = nn.Linear(2048, 20)  # 10 tuổi * 2 giới tính = 20 lớp

    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    if args.prepare_checkpoint_path:
        checkpoint = torch.load(args.prepare_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
    else:
        start_epoch = 0
        best_acc = -1

    age_mapping = ['16-20', '21-25', '26-30', '31-35', '36-40',
                   '41-45', '46-50', '51-55', '56-60', '61-70']
    class_names = [f'{age}-{gender}' for age in age_mapping for gender in ['Female', 'Male']]

    for epoch in range(start_epoch, args.num_epoch):
        model.train()
        progress_bar = tqdm(train_dataloader)
        for iter, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f'Epoch: {epoch}/{args.num_epoch}. Loss: {loss.item():.4f}')
            writer.add_scalar("Train/Loss", loss, iter + epoch * len(train_dataloader))

        model.eval()
        all_labels, all_predictions, all_losses = [], [], []
        with torch.no_grad():
            progress_bar = tqdm(val_dataloader)
            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images)
                loss = criterion(predictions, labels)
                predictions = torch.argmax(predictions, dim=1)
                all_labels.extend(labels.tolist())
                all_predictions.extend(predictions.tolist())
                all_losses.append(loss.item())

        acc = accuracy_score(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, all_predictions)
        loss = np.mean(all_losses)

        precision_per_class = precision_score(all_labels, all_predictions, average=None, labels=range(len(class_names)))
        recall_per_class = recall_score(all_labels, all_predictions, average=None, labels=range(len(class_names)))

        writer.add_scalar('Val/Accuracy', acc, epoch)
        writer.add_scalar('Val/Loss', loss, epoch)
        for idx, class_name in enumerate(class_names):
            writer.add_scalar(f'Val/Precision_{class_name}', precision_per_class[idx], epoch)
            writer.add_scalar(f'Val/Recall_{class_name}', recall_per_class[idx], epoch)
        plot_confusion_matrix(writer, cm, class_names, epoch)

        checkpoint = {'epoch': epoch + 1, 'best_acc': best_acc, 'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, os.path.join(args.checkpoint_path, 'last.pt'))
        if acc > best_acc:
            best_acc = acc
            torch.save(checkpoint, os.path.join(args.checkpoint_path, 'best.pt'))


if __name__ == '__main__':
    args = get_args()
    train(args)