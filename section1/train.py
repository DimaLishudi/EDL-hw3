import torch
from torch import nn
from tqdm.auto import tqdm

from section1.unet import Unet
from section1.dataset import get_train_data
from typing import List

class GradScaler():
    def __init__(
            self,
            scaling_type: str="None",
            loss_scale: float=2**16,
            up_scale: float=2,
            down_scale: float=2,
            up_scale_freq: int=5,
        ):
        self.scaling_type = scaling_type
        self.loss_scale = loss_scale
        self.up_scale = up_scale
        self.down_scale = down_scale
        self.up_scale_freq = up_scale_freq
        self.loss_scale_list = []


    def step(self, optimizer, loss):
        if self.scaling_type != "const" and self.scaling_type != "dynamic":
            loss.backward()
            optimizer.step()
            return

        if self.scaling_type =="dynamic":
            self.loss_scale_list.append(self.loss_scale)
        loss *= self.loss_scale
        loss.backward()
        with torch.no_grad():
            for p_group in optimizer.param_groups:
                for p in p_group["params"]:
                    if not torch.isfinite(p.grad).all():
                        if self.scaling_type == "dynamic":
                            good_grads_counter = 0
                            loss_scale /= self.down_scale
                        return
                    p.grad /= self.loss_scale
        if self.scaling_type == "dynamic":
            good_grads_counter += 1
            if good_grads_counter == self.up_scale_freq:
                self.loss_scale *= self.up_scale
                good_grads_counter = 0
        optimizer.step()
        loss /= self.loss_scale
        

def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
) -> None:   
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.step(optimizer, loss)

        accuracy = ((outputs > 0.5) == labels).float().mean()

        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")


def train(seed=0, **kwargs):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data()
    scaler = GradScaler(**kwargs)
    num_epochs = 5
    for epoch in range(0, num_epochs):
        train_epoch(train_loader, model, criterion, optimizer, device=device, scaler=scaler)
    return scaler.loss_scale_list
