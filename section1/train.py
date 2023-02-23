import torch
from torch import nn
from tqdm.auto import tqdm

from section1.unet import Unet
from section1.dataset import get_train_data
from typing import List

# https://pytorch.org/docs/stable/amp.html#gradient-scaling
# добавил возращение списка коэффициентов скалирования лосса для динамического скалирования
def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaling_type: str="None",
    loss_scale: float=2**16,
    up_scale: float=2,
    down_scale: float=2,
    up_scale_freq: int=2000,
) -> List[float]:   
    model.train()
    loss_scale_list = []

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        if scaling_type == "const" or scaling_type == "dynamic":
            loss *= loss_scale
            loss.backward()
            bad_grads = False
            with torch.no_grad():
                for p_group in optimizer.param_groups:
                    for p in p_group["params"]:
                        if not torch.isfinite(p.grad).all():
                            bad_grads = True
                            break
                        p.grad /= loss_scale
                    if bad_grads:
                        break
            if scaling_type == "dynamic":
                loss_scale_list.append(loss_scale)
                if bad_grads:
                    loss_scale /= down_scale
                else:
                    good_grad_counter += 1
                    if good_grad_counter == up_scale_freq:
                        loss_scale *= up_scale
                        good_grad_counter = 0
            if not bad_grads:
                optimizer.step()
            loss /= loss_scale
        else:
            loss.backward()
            optimizer.step()
        print(torch.cuda.memory_reserved(0))

        accuracy = ((outputs > 0.5) == labels).float().mean()

        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")
    return loss_scale_list

def train(**kwargs):
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data()
    scales = []
    num_epochs = 5
    for epoch in range(0, num_epochs):
        scales.append(train_epoch(train_loader, model, criterion, optimizer, device=device, **kwargs))
    return scales
