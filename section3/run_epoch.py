import typing as tp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from section3 import dataset
from section3.utils import Settings
from section3.vit import ViT as SubOptimalViT

from torch.profiler import profile, ProfilerActivity


def get_vit_model() -> torch.nn.Module:
    model = SubOptimalViT(
        dim=128,
        mlp_dim=128,
        depth=12,
        heads=8,
        image_size=224,
        patch_size=32,
        num_classes=2,
        channels=3,
    ).to(Settings.device)
    return model


def get_train_loader() -> torch.utils.data.DataLoader:
    train_list = dataset.extract_dataset_globs(half=False)
    print(f"Train Data: {len(train_list)}")
    train_transforms = dataset.get_train_transforms()
    train_data = dataset.CatsDogsDataset(train_list, transform=train_transforms)
    train_loader = DataLoader(dataset=train_data, batch_size=Settings.batch_size, shuffle=True)

    return train_loader


def run_epoch(model, train_loader, criterion, optimizer) -> tp.Tuple[float, float]:
    epoch_loss, epoch_accuracy = 0, 0
    
    max_steps = len(train_loader) // 5
    for i, (data, label) in tqdm(enumerate(train_loader), desc=f"[Train]", total=max_steps):
        data = data.to(Settings.device)
        label = label.to(Settings.device)
        with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    profile_memory=True, record_shapes=True) as forward_prof:
            output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()

        with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    profile_memory=True, record_shapes=True) as backward_prof:
            loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
        if i >= max_steps:
            break
    return epoch_loss, epoch_accuracy, forward_prof, backward_prof


def warmup(model, train_loader):
    for i, (data, label) in enumerate(train_loader):
        data = data.to(Settings.device)
        label = label.to(Settings.device)
        model(data)
        if i >= 2:
            break


def main():
    model = get_vit_model()
    train_loader = get_train_loader()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Settings.lr)

    warmup(model, train_loader)
    return run_epoch(model, train_loader, criterion, optimizer)


if __name__ == "__main__":
    main()
